"""Format / Progress / Outcome scorer for tree trajectories.

Direct port of ISP-KGR/k_beam_score.py. The synchronous `requests.post` calls
to the entity-distance API are replaced by `await kg_client.entity_distance`.
Everything else (sub-score weights 0.1/0.3/0.6, variable tracking, prev_node
state machine, max-aggregation across trajectories) is kept verbatim so the
training signal matches your published recipe.

Entry points:
- `compute_distance_to_answer(entities, gt, kg_client, max_distance)`
- `score_trajectory_path(path_ids, trajectory_tree, gt, trajectory_final_f1,
   kg_client, max_distance) -> list[step_score_dict]`
- `score_tree(trajectory_tree, sample_summaries, kg_client, max_distance)
   -> dict[node_id, reward_float]`
"""

from __future__ import annotations

import ast
import logging
import re
from collections import defaultdict
from typing import Any

from .kg_client import KGClient
from .kg_utils import (
    F1_score,
    check_sparql_syntax,
    check_xml_enter,
    is_failure,
    validate_xml_tags,
)

logger = logging.getLogger(__name__)


def extract_node_content(output_text: str) -> str | None:
    m = re.search(r"<node>\n\s*(.*?)\s*</node>", output_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_sparql_from_text(output_text: str) -> str | None:
    m = re.search(r"<sparql>\n\s*(.*?)\s*</sparql>", output_text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_mention_entity(prompt: str):
    """Lift the mention-entity hint out of the original prompt template tail."""
    try:
        tail = (
            prompt.split(
                "The standard name of the entity involved in the question in the knowledge graph is:\n"
            )[-1]
            .split("<|im_end|>")[0]
            .strip()
        )
        if tail:
            return ast.literal_eval(tail)
    except Exception:
        return []
    return []


def extract_select_variable(sparql: str) -> str | None:
    m = re.search(r"SELECT\s+DISTINCT\s+(\?\w+)", sparql, re.IGNORECASE)
    return m.group(1) if m else None


def flatten_sparql_result(query_result: Any) -> list[str]:
    if not query_result:
        return []
    if isinstance(query_result, list):
        if not query_result:
            return []
        if isinstance(query_result[0], str):
            return query_result
        if isinstance(query_result[0], dict):
            out: list[str] = []
            for item in query_result:
                for v in item.values():
                    out.append(str(v))
            return out
    return []


async def compute_distance_to_answer(
    entities: list[str],
    ground_truth: list[str],
    kg_client: KGClient,
    max_distance: int = 3,
) -> float:
    """Returns a score in [0, 1] where higher = closer to ground truth.

    Matches k_beam_score.py:86-127 exactly:
    - F1 >= 0.9 → 1.0
    - F1 > 0    → 0.5 + 0.5 * F1
    - Else      → distance-API normalized score (max_distance - d) / max_distance
    """
    if not entities or not ground_truth:
        return 0.0

    try:
        f1 = F1_score(entities, ground_truth)
        if f1 >= 0.9:
            return 1.0
        if f1 > 0:
            return 0.5 + f1 * 0.5
    except Exception:
        pass

    distance = await kg_client.entity_distance(
        set_a=list(entities),
        set_b=list(ground_truth),
        max_distance=max_distance,
        early_stop_global_min=1,
    )
    if distance is None:
        return 0.0
    return (max_distance - distance) / max_distance


async def score_trajectory_path(
    path_ids: list[str],
    trajectory_tree: dict[str, dict],
    ground_truth: list[str],
    trajectory_final_f1: float,
    kg_client: KGClient,
    max_distance: int = 3,
) -> list[dict]:
    """Score every node in one root-to-leaf trajectory path.

    Returns a list of per-step dicts with keys:
    `trajectory_id, step, output, query_type, progress_score, format_score,
    outcome_score, total_score, current_distance`.

    total_score = 0.1 * format + 0.3 * progress + 0.6 * outcome
    """
    if not path_ids:
        return []

    root = trajectory_tree[path_ids[0]]
    mention_entity = extract_mention_entity(root.get("prompt", ""))
    mention_entity_score = 0.0
    if mention_entity:
        mention_entity_score = await compute_distance_to_answer(
            [mention_entity] if not isinstance(mention_entity, list) else mention_entity,
            ground_truth,
            kg_client,
            max_distance,
        )

    variable_scores: dict[str, float] = {}
    executed_sparqls: set[str] = set()
    prev_node_state = {"type": "START", "score": 0.0}
    step_scores: list[dict] = []

    for node_id in path_ids:
        if node_id not in trajectory_tree:
            continue
        tree_node = trajectory_tree[node_id]
        output_text = tree_node.get("output", "")
        query_type = tree_node.get("query_type")

        # NOTE: ISP-KGR/k_beam_score.py:178-181 re-appends the closing tag
        # because VLLM (default `include_stop_str_in_output=False`) strips it.
        # Slime/SGLang here runs with `no_stop_trim=True`, so output_text
        # ALREADY contains the closing tag and we must not re-append.

        # ----- Format -----
        has_think = "<think>" in output_text.lower() and "</think>" in output_text.lower()
        if not has_think:
            format_score = -1.0
        elif not validate_xml_tags(output_text):
            format_score = -1.0
        elif not check_xml_enter(output_text)[0]:
            format_score = -1.0
        else:
            format_score = 1.0

        # ----- Progress -----
        progress_score = 0.0
        current_node_type = "OTHER"
        current_node_score = 0.0

        if query_type == "node":
            current_node_type = "NODE"
            node_content = extract_node_content(output_text)

            is_mention = (
                mention_entity
                and node_content
                and (
                    node_content in mention_entity
                    if isinstance(mention_entity, (list, tuple))
                    else node_content == mention_entity
                )
            )
            is_variable = node_content in variable_scores if node_content else False

            if is_mention:
                progress_score = 0.0
                current_node_score = mention_entity_score
            elif is_variable:
                progress_score = 0.0
                current_node_score = variable_scores[node_content]
            else:
                progress_score = -1.0
                current_node_score = 0.0

        elif query_type == "sparql":
            current_node_type = "SPARQL"
            sparql = extract_sparql_from_text(output_text)

            if not sparql or not check_sparql_syntax(sparql):
                progress_score = -1.0
            elif sparql in executed_sparqls:
                progress_score = -1.0
            else:
                executed_sparqls.add(sparql)

                query_result = tree_node.get("query_result", [])
                if is_failure(query_result):
                    # ISP-KGR `continue`-skips this node entirely
                    continue
                if isinstance(query_result, str):
                    try:
                        query_result = ast.literal_eval(query_result)
                    except Exception:
                        query_result = []
                flat_result = flatten_sparql_result(query_result)
                result_score = await compute_distance_to_answer(
                    flat_result, ground_truth, kg_client, max_distance
                )
                current_node_score = result_score

                selected_var = extract_select_variable(sparql)
                if selected_var:
                    old = variable_scores.get(selected_var, 0.0)
                    variable_scores[selected_var] = max(old, result_score)

                if prev_node_state["type"] == "NODE":
                    progress_score = 1.0 if result_score > prev_node_state["score"] else 0.0
                else:
                    f1 = F1_score(flat_result, ground_truth) if flat_result else 0.0
                    progress_score = 1.0 if f1 >= 0.9 else 0.0

        elif "</answer>" in output_text:
            current_node_type = "ANSWER"
            progress_score = 0.0
        else:
            progress_score = 0.0

        # ----- Outcome -----
        if format_score == -1.0 or progress_score == -1.0:
            outcome_score = 0.0
        else:
            outcome_score = trajectory_final_f1

        total_score = 0.1 * format_score + 0.3 * progress_score + 0.6 * outcome_score

        if current_node_type == "NODE" and progress_score != -1.0:
            prev_node_state = {"type": "NODE", "score": current_node_score}
        elif current_node_type == "SPARQL" and progress_score != -1.0:
            prev_node_state = {"type": "SPARQL", "score": current_node_score}
        elif progress_score == -1.0:
            prev_node_state = {"type": "INVALID", "score": 0.0}

        step_scores.append(
            {
                "trajectory_id": node_id,
                "step": tree_node.get("step", 0),
                "output": output_text,
                "query_type": query_type,
                "progress_score": progress_score,
                "format_score": format_score,
                "outcome_score": outcome_score,
                "total_score": round(total_score, 4),
                "current_distance": current_node_score,
            }
        )

    return step_scores


async def score_tree(
    trajectory_tree: dict[str, dict],
    sample_summaries_for_question: list[dict],
    ground_truth: list[str],
    kg_client: KGClient,
    max_distance: int = 3,
) -> dict[str, float]:
    """Aggregate per-node rewards across all trajectories passing through them.

    Args:
        trajectory_tree: full tree for one question, node_id -> tree_node dict.
        sample_summaries_for_question: list of trajectory summaries from rollout
            (each has `path`, `predict`, `is_correct`, etc.).
        ground_truth: list[str] of golden answers for this question.

    Returns:
        node_id -> aggregated reward (max over all trajectories passing through).
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    node_scores: dict[str, list[float]] = defaultdict(list)

    for traj in sample_summaries_for_question:
        predict = traj.get("predict", "")
        if predict:
            if isinstance(predict, str):
                predict = [predict]
            trajectory_f1 = F1_score(predict, ground_truth)
        else:
            trajectory_f1 = 0.0

        step_scores = await score_trajectory_path(
            path_ids=traj["path"],
            trajectory_tree=trajectory_tree,
            ground_truth=ground_truth,
            trajectory_final_f1=trajectory_f1,
            kg_client=kg_client,
            max_distance=max_distance,
        )
        for step in step_scores:
            node_scores[step["trajectory_id"]].append(step["total_score"])

    # ISP-KGR k_beam_score.py:399-404 uses MAX aggregation, not mean
    return {node_id: max(scores) for node_id, scores in node_scores.items()}
