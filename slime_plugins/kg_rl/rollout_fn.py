"""K-beam tree rollout for KG-RL, ported from ISP-KGR/infer_reasoning_k_beam.py
into slime's `--rollout-function-path` interface.

Plug-in path (CLI):
    --rollout-function-path slime_plugins.kg_rl.rollout_fn:k_beam_tree_rollout

Semantics:
- Each question expands into a tree: at every active node, sample K candidates,
  dedup by (output, query_result), apply KG queries, recurse for up to
  `max_interactions` rounds.
- After the tree is built, score every node (via `reward_fn.score_tree`).
- Every non-leaf parent contributes ONE training group: its kept children are
  the K completions, with `Sample.reward` already filled in.
- `group_resize.resize_to_k` pads (cyclic) or shrinks (reward-stratified) each
  group to exactly `args.n_samples_per_prompt`, which is what slime requires.
- Slime's `--dynamic-sampling-filter-path
  slime.rollout.filter_hub.dynamic_sampling_filters:check_reward_nonzero_std`
  drops groups whose rewards have zero variance (no GRPO signal).

Custom CLI args read from `args`/env:
- `args.kg_query_url`         (env: KG_QUERY_URL)   default http://localhost:5501/kg_query
- `args.distance_api_url`     (env: KG_DISTANCE_URL) default http://localhost:5501/entity_distance
- `args.kg_beam_candidates`   (env: KG_BEAM_CANDIDATES) default 16
- `args.kg_max_interactions`  (env: KG_MAX_INTERACTIONS) default 6
- `args.kg_max_prompt_tokens` (env: KG_MAX_PROMPT_TOKENS) default 4096
- `args.kg_max_distance`      (env: KG_MAX_DISTANCE) default 3
- `args.kg_per_step_max_tokens` (env: KG_PER_STEP_MAX_TOKENS) default 1024
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.misc import load_function
from slime.utils.types import Sample

from .group_resize import resize_to_k
from .kg_client import KGClient
from .kg_utils import extract_boxed_content, extract_outer_braces_content
from .reward_fn import score_tree

logger = logging.getLogger(__name__)


STOP_TOKENS = ["</sparql>", "</SPARQL>", "</node>"]
RELATION_FORMAT = "{prompt}{stop_reason}\n\n<relation>\n{search_results}</relation>\n\n"
INFORMATION_FORMAT = "{prompt}{stop_reason}\n\n<information>\n{search_results}</information>\n\n"
NO_NODE_RELATION_MSG = "No relevant relations found for the given node query."
NO_SPARQL_RESULT_MSG = (
    "No results found for the given SPARQL query.\nPlease try generating a different SPARQL query."
)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _cfg(args: Namespace) -> dict:
    """Read KG-RL knobs from args (preferred) with env-var fallback."""
    return {
        "kg_query_url": getattr(args, "kg_query_url", None)
            or _env_str("KG_QUERY_URL", "http://localhost:5501/kg_query"),
        "distance_api_url": getattr(args, "distance_api_url", None)
            or _env_str("KG_DISTANCE_URL", "http://localhost:5501/entity_distance"),
        "k_beam_candidates": getattr(args, "kg_beam_candidates", None)
            or _env_int("KG_BEAM_CANDIDATES", 16),
        "max_interactions": getattr(args, "kg_max_interactions", None)
            or _env_int("KG_MAX_INTERACTIONS", 6),
        "max_prompt_tokens": getattr(args, "kg_max_prompt_tokens", None)
            or _env_int("KG_MAX_PROMPT_TOKENS", 4096),
        "max_distance": getattr(args, "kg_max_distance", None) or _env_int("KG_MAX_DISTANCE", 3),
        "per_step_max_tokens": getattr(args, "kg_per_step_max_tokens", None)
            or _env_int("KG_PER_STEP_MAX_TOKENS", 1024),
    }


# -----------------------------------------------------------------------------
# Tree data structures (one per question)
# -----------------------------------------------------------------------------

@dataclass
class _TreeNode:
    trajectory_id: str
    parent_id: str | None
    sample_index: int
    step: int
    prompt: str                 # multi-turn context up to this node
    output: str = ""            # one-step model output (between stop tokens)
    stop_reason: str | None = None
    query_type: str | None = None   # "node" | "sparql" | None
    query_info: dict | None = None
    query_result: Any = None    # list or str
    query_result_str: str = ""
    children: list[str] = field(default_factory=list)
    is_complete: bool = False
    predict: str = ""
    updated_prompt: str | None = None  # prompt fed to next-step generation


@dataclass
class _ActiveState:
    """Walking-pointer into the tree during expansion."""
    trajectory_id: str
    parent_id: str | None
    sample_index: int
    prompt: str
    previous_sparql: str
    cnt: int
    question: str
    ground_truth: list[str]
    complete: bool
    step: int


@dataclass
class _Candidate:
    parent_trajectory_id: str
    candidate_index: int
    output: str
    stop_reason: str | None
    query_type: str | None
    query_info: dict | None
    is_complete: bool
    predict: str
    query_result: Any = None
    query_result_str: str = ""


# -----------------------------------------------------------------------------
# SGLang single-shot generation
# -----------------------------------------------------------------------------

def _detect_stop_suffix(text: str) -> tuple[str | None, str | None]:
    """Return (matched_stop_token, normalized_query_type) by suffix-match."""
    stripped = text.rstrip()
    for tok in STOP_TOKENS:
        if stripped.endswith(tok):
            qtype = "node" if tok == "</node>" else "sparql"
            return tok, qtype
    return None, None


async def _sglang_generate(
    args: Namespace,
    state: GenerateState,
    prompt_text: str,
    sampling_params: dict[str, Any],
) -> dict:
    """One SGLang `/generate` call. Returns the full response dict.
    Concurrency-bounded by `state.semaphore`."""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    prompt_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": False,
    }
    async with state.semaphore:
        if state.aborted:
            return {"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}
        with state.dp_rank_context() as _:
            return await post(url, payload)


def _parse_candidate(
    response: dict,
    parent_id: str,
    candidate_index: int,
    previous_sparql: str,
    question: str,
) -> _Candidate:
    output = response.get("text", "")
    finish_reason = response.get("meta_info", {}).get("finish_reason", {}).get("type", "")

    stop_reason, query_type = (None, None)
    if finish_reason == "stop":
        stop_reason, query_type = _detect_stop_suffix(output)

    cand = _Candidate(
        parent_trajectory_id=parent_id,
        candidate_index=candidate_index,
        output=output,
        stop_reason=stop_reason,
        query_type=query_type,
        query_info=None,
        is_complete=False,
        predict="",
    )

    if finish_reason == "abort":
        cand.is_complete = True
        return cand

    if query_type == "node":
        identify_node = output.split("<node>")[-1].split("</node>")[0].strip()
        cand.query_info = {
            "identify_node": identify_node,
            "query_content": question,
            "sparql": previous_sparql,
        }
    elif query_type == "sparql":
        # Mirror infer_reasoning_k_beam.py:342-347 — case-insensitive split on <sparql>
        parts = re.split(r"<sparql>", output, flags=re.IGNORECASE)
        step_sparql = parts[-1].split("</sparql>")[0].split("</SPARQL>")[0].strip()
        cand.query_info = {"step_sparql": step_sparql}
    else:
        # No stop matched, or stopped without a recognized query tag → done
        cand.is_complete = True
        boxed = extract_boxed_content(output)
        cand.predict = boxed if boxed else ""

    return cand


# -----------------------------------------------------------------------------
# Dedup + KG execution (per round)
# -----------------------------------------------------------------------------

async def _execute_kg_queries(
    candidates: list[_Candidate],
    kg_client: KGClient,
) -> None:
    """Run KG queries with dedup: identical (type, content) only fires once.
    Fill `query_result` and `query_result_str` in-place."""
    import json

    query_map: dict[tuple, list[_Candidate]] = defaultdict(list)
    for cand in candidates:
        if cand.query_type is None:
            continue
        if cand.query_type == "node":
            info = cand.query_info
            key = ("node", info["identify_node"], info["sparql"], info["query_content"])
        elif cand.query_type == "sparql":
            info = cand.query_info
            key = ("sparql", info["step_sparql"])
        else:
            continue
        query_map[key].append(cand)

    if not query_map:
        return

    async def _run_one(key: tuple, group: list[_Candidate]):
        cand = group[0]
        try:
            if cand.query_type == "node":
                info = cand.query_info
                identify_node = info["identify_node"]
                if identify_node.startswith("m.") or identify_node.startswith("g."):
                    result = await kg_client.query_node_relation(
                        mid=identify_node, sparql="", question=info["query_content"]
                    )
                elif identify_node.startswith("?"):
                    if identify_node in info["sparql"]:
                        result = await kg_client.query_node_relation(
                            mid=identify_node,
                            sparql=info["sparql"],
                            question=info["query_content"],
                        )
                    else:
                        result = []
                elif identify_node:
                    result = await kg_client.query_node_relation(
                        mid=identify_node, sparql="", question=info["query_content"]
                    )
                else:
                    result = []
                result_str = json.dumps(result, sort_keys=True)
            else:  # sparql
                result = await kg_client.query_full_sparql(cand.query_info["step_sparql"])
                result_str = json.dumps(result, sort_keys=True)
        except Exception as e:
            logger.warning("KG query failed for %s: %s", key[:2], e)
            result, result_str = [], "[]"
        for c in group:
            c.query_result = result
            c.query_result_str = result_str

    await asyncio.gather(*(_run_one(k, g) for k, g in query_map.items()))


def _dedup_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """Keep first-seen of each (uniqueness key). Mirrors
    infer_reasoning_k_beam.py:482-524."""
    seen: set = set()
    kept: list[_Candidate] = []
    for cand in candidates:
        is_failed_query = cand.query_type and cand.query_result_str == "[]"
        if is_failed_query:
            key = ("failed_query", cand.output)
        elif cand.query_type:
            key = ("successful_query", cand.query_type, cand.query_result_str)
        elif cand.is_complete:
            key = ("complete", cand.output)
        else:
            key = ("other", cand.output)
        if key in seen:
            continue
        seen.add(key)
        kept.append(cand)
    return kept


# -----------------------------------------------------------------------------
# Tree expansion (one question)
# -----------------------------------------------------------------------------

async def _expand_tree(
    args: Namespace,
    state: GenerateState,
    kg_client: KGClient,
    root_prompt: str,
    question: str,
    ground_truth: list[str],
    sample_index: int,
    cfg: dict,
) -> dict[str, _TreeNode]:
    """Build the tree for one question. Returns trajectory_id -> _TreeNode."""

    tree: dict[str, _TreeNode] = {}
    root_id = f"sample_{sample_index}_step_0"
    tree[root_id] = _TreeNode(
        trajectory_id=root_id,
        parent_id=None,
        sample_index=sample_index,
        step=0,
        prompt=root_prompt,
    )
    active: list[_ActiveState] = [
        _ActiveState(
            trajectory_id=root_id,
            parent_id=None,
            sample_index=sample_index,
            prompt=root_prompt,
            previous_sparql="",
            cnt=0,
            question=question,
            ground_truth=ground_truth,
            complete=False,
            step=0,
        )
    ]

    sampling_params = state.sampling_params.copy()
    sampling_params["max_new_tokens"] = cfg["per_step_max_tokens"]
    sampling_params["stop"] = STOP_TOKENS
    sampling_params["no_stop_trim"] = True

    for round_idx in range(cfg["max_interactions"]):
        if not active:
            break

        # 1) For each active state, fire K parallel generations
        gen_tasks = []
        slot_map: list[tuple[int, int]] = []  # (state_idx, k_idx)
        for state_idx, st in enumerate(active):
            for k_idx in range(cfg["k_beam_candidates"]):
                gen_tasks.append(_sglang_generate(args, state, st.prompt, sampling_params))
                slot_map.append((state_idx, k_idx))

        responses = await asyncio.gather(*gen_tasks)

        # 2) Parse into candidates, group by parent
        candidates_by_parent: dict[str, list[_Candidate]] = defaultdict(list)
        for resp_idx, resp in enumerate(responses):
            state_idx, k_idx = slot_map[resp_idx]
            parent_state = active[state_idx]
            cand = _parse_candidate(
                resp,
                parent_id=parent_state.trajectory_id,
                candidate_index=k_idx,
                previous_sparql=parent_state.previous_sparql,
                question=parent_state.question,
            )
            candidates_by_parent[parent_state.trajectory_id].append(cand)

        # 3) Execute KG queries (with cross-parent dedup)
        all_candidates = [c for cands in candidates_by_parent.values() for c in cands]
        await _execute_kg_queries(all_candidates, kg_client)

        # 4) Build next-round active list
        next_active: list[_ActiveState] = []
        child_counter: dict[str, int] = defaultdict(int)

        for st in active:
            parent_id = st.trajectory_id
            cands = candidates_by_parent.get(parent_id, [])
            kept = _dedup_candidates(cands)

            for cand in kept:
                child_id = f"{parent_id}_child_{child_counter[parent_id]}"
                child_counter[parent_id] += 1

                new_prompt = st.prompt
                new_previous_sparql = st.previous_sparql
                new_cnt = st.cnt
                child_complete = cand.is_complete
                child_predict = cand.predict
                updated_prompt: str | None = None

                if cand.is_complete:
                    new_prompt = st.prompt + cand.output

                elif cand.query_type == "node":
                    relations = cand.query_result or []
                    new_cnt += 1
                    if relations:
                        updated_prompt = RELATION_FORMAT.format(
                            prompt=st.prompt + cand.output,
                            stop_reason=cand.stop_reason or "",
                            search_results="\n".join(relations),
                        )
                        new_prompt = updated_prompt
                    else:
                        updated_prompt = RELATION_FORMAT.format(
                            prompt=st.prompt + cand.output,
                            stop_reason=cand.stop_reason or "",
                            search_results=NO_NODE_RELATION_MSG,
                        )
                        new_prompt = updated_prompt
                        child_complete = True

                elif cand.query_type == "sparql":
                    research_info = cand.query_result or []
                    new_cnt += 1
                    if research_info:
                        new_previous_sparql = extract_outer_braces_content(
                            cand.query_info["step_sparql"]
                        )
                        updated_prompt = INFORMATION_FORMAT.format(
                            prompt=st.prompt + cand.output,
                            stop_reason=cand.stop_reason or "",
                            search_results=research_info,
                        )
                        new_prompt = updated_prompt
                    else:
                        updated_prompt = INFORMATION_FORMAT.format(
                            prompt=st.prompt + cand.output,
                            stop_reason=cand.stop_reason or "",
                            search_results=NO_SPARQL_RESULT_MSG,
                        )
                        new_prompt = updated_prompt
                        child_complete = True

                # Length cap
                tokens = state.tokenizer(new_prompt, add_special_tokens=False)["input_ids"]
                if len(tokens) >= cfg["max_prompt_tokens"]:
                    child_complete = True

                tree[child_id] = _TreeNode(
                    trajectory_id=child_id,
                    parent_id=parent_id,
                    sample_index=st.sample_index,
                    step=round_idx + 1,
                    prompt=st.prompt,        # child's *training* prompt = parent's prompt
                    output=cand.output,
                    stop_reason=cand.stop_reason,
                    query_type=cand.query_type,
                    query_info=cand.query_info,
                    query_result=cand.query_result,
                    query_result_str=cand.query_result_str,
                    children=[],
                    is_complete=child_complete,
                    predict=child_predict,
                    updated_prompt=updated_prompt,
                )
                tree[parent_id].children.append(child_id)

                if not child_complete:
                    next_active.append(
                        _ActiveState(
                            trajectory_id=child_id,
                            parent_id=parent_id,
                            sample_index=st.sample_index,
                            prompt=new_prompt,
                            previous_sparql=new_previous_sparql,
                            cnt=new_cnt,
                            question=st.question,
                            ground_truth=st.ground_truth,
                            complete=False,
                            step=round_idx + 1,
                        )
                    )

        active = next_active

    return tree


# -----------------------------------------------------------------------------
# Tree → list of training samples
# -----------------------------------------------------------------------------

def _build_sample_summaries_for_tree(
    tree: dict[str, _TreeNode],
) -> list[dict]:
    """Walk leaves → emit per-trajectory summary dicts needed by score_tree."""
    summaries: list[dict] = []
    for node_id, node in tree.items():
        if node.children:
            continue
        # build path from root to this leaf
        path: list[str] = []
        cur = node_id
        while cur is not None:
            path.insert(0, cur)
            cur = tree[cur].parent_id if cur in tree else None
        leaf = tree[node_id]
        summaries.append(
            {
                "path": path,
                "predict": leaf.predict,
                "is_complete": leaf.is_complete,
            }
        )
    return summaries


def _trailing_for(query_type: str | None, output: str) -> str:
    """Suffix to append to the SGLang output before training.

    With `no_stop_trim=True` (slime default), `</node>`/`</sparql>` are already
    in `output`. For a *complete* answer trajectory we additionally append the
    chat-template's end-of-turn marker so the model is trained to stop there
    (matches k_beam_score.py:411 which appends `<|im_end|>`)."""
    if query_type in ("node", "sparql"):
        return ""
    if "\\boxed{" in output or "//boxed{" in output:
        return "<|im_end|>"
    return ""


def _node_to_sample(
    parent_node: _TreeNode,
    child_node: _TreeNode,
    reward: float,
    ground_truth: list[str],
    tokenizer,
    group_index: int,
    sample_index: int,
) -> Sample:
    """Build one training Sample from a (parent, child) pair."""
    response_str = child_node.output + _trailing_for(child_node.query_type, child_node.output)

    # Tokenize the full (prompt + response) text together to avoid BPE merge
    # mismatches across the boundary. The response slice is the trailing region
    # whose length equals `full_tokens - prompt_tokens`.
    prompt_ids = tokenizer(parent_node.prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        parent_node.prompt + response_str, add_special_tokens=False
    )["input_ids"]
    response_length = max(0, len(full_ids) - len(prompt_ids))
    tokens = list(full_ids)
    # slime convention (slime/ray/rollout.py:670): loss_mask length must equal
    # response_length — it covers the response slice only, not the prompt.
    loss_mask = [1] * response_length

    # Every emitted sample is already a completed one-step generation (the tree
    # rollout has finished by the time we slice into training samples); the
    # `is_complete` flag on `child_node` only signals whether the *trajectory*
    # ends here, not whether this generation completed.
    status = Sample.Status.COMPLETED

    return Sample(
        group_index=group_index,
        index=sample_index,
        prompt=parent_node.prompt,
        response=response_str,
        response_length=response_length,
        tokens=tokens,
        loss_mask=loss_mask,
        label=list(ground_truth) if ground_truth else None,
        reward=float(reward),
        status=status,
        metadata={
            "child_trajectory_id": child_node.trajectory_id,
            "parent_trajectory_id": parent_node.trajectory_id,
            "child_step": child_node.step,
            "query_type": child_node.query_type,
            "is_complete": child_node.is_complete,
            "predict": child_node.predict,
            "sample_index_in_tree": child_node.sample_index,
        },
    )


def _slice_tree_into_groups(
    tree: dict[str, _TreeNode],
    node_rewards: dict[str, float],
    ground_truth: list[str],
    tokenizer,
    n_per_group: int,
    rng: random.Random,
    next_index: list[int],          # mutable one-element list = global counter
    next_group_index: list[int],
) -> list[list[Sample]]:
    """For every non-leaf parent, emit one group of n_per_group Samples."""
    groups: list[list[Sample]] = []

    parent_ids = [nid for nid, node in tree.items() if node.children]
    for parent_id in parent_ids:
        parent_node = tree[parent_id]
        child_ids = parent_node.children
        # Collect kept (= all children — dedup was done at expansion time)
        items: list[_TreeNode] = []
        rewards: list[float] = []
        for cid in child_ids:
            child = tree.get(cid)
            if child is None:
                continue
            items.append(child)
            rewards.append(round(node_rewards.get(cid, 0.0), 4))

        if not items:
            continue

        items, rewards = resize_to_k(items, rewards, n_per_group, rng)

        gi = next_group_index[0]
        next_group_index[0] += 1
        group_samples: list[Sample] = []
        for child, r in zip(items, rewards):
            sample = _node_to_sample(
                parent_node=parent_node,
                child_node=child,
                reward=r,
                ground_truth=ground_truth,
                tokenizer=tokenizer,
                group_index=gi,
                sample_index=next_index[0],
            )
            next_index[0] += 1
            group_samples.append(sample)
        groups.append(group_samples)

    return groups


# -----------------------------------------------------------------------------
# Top-level orchestration
# -----------------------------------------------------------------------------

async def _k_beam_tree_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source,
) -> RolloutFnTrainOutput:
    cfg = _cfg(args)
    state = GenerateState(args)
    rng = random.Random(args.rollout_seed + rollout_id)

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None) is not None
        else None
    )

    metric_gatherer = MetricGatherer()
    target_data_size = args.rollout_batch_size

    data: list[list[Sample]] = []
    next_index = [0]
    next_group_index = [rollout_id * 10_000_000]  # avoid collisions across rollouts

    async with KGClient(
        kg_query_url=cfg["kg_query_url"],
        distance_api_url=cfg["distance_api_url"],
    ) as kg_client:

        # Pull questions in small batches and expand trees in parallel. A single
        # tree typically emits several training groups (one per non-leaf
        # parent), so we don't need 1:1 questions → groups.
        questions_per_batch = max(1, target_data_size // 8)
        while len(data) < target_data_size:
            # data_source.get_samples returns list[list[Sample]]; the inner
            # lists are `n_samples_per_prompt` identical copies of one prompt,
            # but our tree creates its own diversity so we only need the first.
            prompt_groups = data_source.get_samples(questions_per_batch)
            if not prompt_groups:
                logger.warning("data_source returned no more samples; stopping early")
                break

            # Expand all trees in parallel
            tree_tasks = []
            metas: list[tuple[Sample, list[str]]] = []
            for pg in prompt_groups:
                if not pg:
                    continue
                root_sample = pg[0]
                gt = root_sample.label or []
                if isinstance(gt, str):
                    gt = [gt]
                tree_tasks.append(
                    _expand_tree(
                        args=args,
                        state=state,
                        kg_client=kg_client,
                        root_prompt=root_sample.prompt if isinstance(root_sample.prompt, str) else "",
                        question=(root_sample.metadata or {}).get("question") or root_sample.prompt,
                        ground_truth=gt,
                        sample_index=root_sample.index or 0,
                        cfg=cfg,
                    )
                )
                metas.append((root_sample, gt))

            trees = await asyncio.gather(*tree_tasks)

            # For each tree: score, slice, filter
            for tree, (root_sample, gt) in zip(trees, metas):
                summaries = _build_sample_summaries_for_tree(tree)
                node_rewards = await score_tree(
                    trajectory_tree={nid: _treenode_to_dict(n) for nid, n in tree.items()},
                    sample_summaries_for_question=summaries,
                    ground_truth=gt,
                    kg_client=kg_client,
                    max_distance=cfg["max_distance"],
                )

                groups = _slice_tree_into_groups(
                    tree=tree,
                    node_rewards=node_rewards,
                    ground_truth=gt,
                    tokenizer=state.tokenizer,
                    n_per_group=args.n_samples_per_prompt,
                    rng=rng,
                    next_index=next_index,
                    next_group_index=next_group_index,
                )

                for group in groups:
                    if len(data) >= target_data_size:
                        break
                    out = call_dynamic_filter(dynamic_filter, args, group)
                    if not out.keep:
                        metric_gatherer.on_dynamic_filter_drop(reason=out.reason)
                        continue
                    data.append(group)

                if len(data) >= target_data_size:
                    break

    if len(data) > target_data_size:
        data = data[:target_data_size]

    if len(data) == 0:
        logger.error("No groups survived rollout/filtering — check data and reward signal")

    metrics = metric_gatherer.collect()
    metrics["rollout/groups_emitted"] = len(data)
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def _treenode_to_dict(node: _TreeNode) -> dict:
    """`score_trajectory_path` expects a dict-shaped node (same fields as the
    original ISP-KGR trajectory_tree)."""
    return {
        "trajectory_id": node.trajectory_id,
        "parent_id": node.parent_id,
        "sample_index": node.sample_index,
        "step": node.step,
        "prompt": node.prompt,
        "output": node.output,
        "stop_reason": node.stop_reason,
        "query_type": node.query_type,
        "query_info": node.query_info,
        "query_result": node.query_result,
        "children": node.children,
        "is_complete": node.is_complete,
        "predict": node.predict,
    }


def k_beam_tree_rollout(
    args: Namespace,
    rollout_id: int,
    data_source,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Synchronous entry point — what slime calls via --rollout-function-path."""
    if evaluation:
        raise NotImplementedError(
            "K-beam tree rollout does not implement evaluation yet. "
            "Set --eval-interval -1 or supply --eval-function-path."
        )
    return asyncio.run(_k_beam_tree_rollout_async(args, rollout_id, data_source))
