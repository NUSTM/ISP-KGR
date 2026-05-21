# KG-RL inside slime (ISP-KGR port)

This example reproduces the **Interactive Semantic Parsing GRPO** training
recipe ([NUSTM/ISP-KGR](https://github.com/NUSTM/ISP-KGR)) on top of slime's
plug-in API. Compared to the original repo, the algorithm is **byte-for-byte
identical**; the framework around it is now slime's Ray + Megatron + SGLang
stack instead of a hand-rolled HF Trainer pipeline.

## What's customised vs. vanilla GRPO

| ISP-KGR feature | slime plug-in point | File |
| --- | --- | --- |
| K-beam **tree** rollout with KG environment (one question → multiple branching parents → many training groups) | `--rollout-function-path` | [`slime_plugins/kg_rl/rollout_fn.py`](../../slime_plugins/kg_rl/rollout_fn.py) |
| Per-node `0.1·format + 0.3·progress + 0.6·outcome` reward, max-aggregated across trajectories | filled inside rollout_fn before emit; uses external `/entity_distance` API | [`slime_plugins/kg_rl/reward_fn.py`](../../slime_plugins/kg_rl/reward_fn.py) |
| Variable group size after candidate dedup → cyclic-copy / reward-stratified resize to fixed `K=6` | called from rollout_fn | [`slime_plugins/kg_rl/group_resize.py`](../../slime_plugins/kg_rl/group_resize.py) |
| Drop groups whose rewards have zero variance | slime built-in via `--dynamic-sampling-filter-path` | `slime.rollout.filter_hub.dynamic_sampling_filters:check_reward_nonzero_std` |
| Asymmetric PPO clip (ε_low=0.2, ε_high=0.24) — DAPO-style | slime native CLI | `--eps-clip 0.2 --eps-clip-high 0.24` |
| Advantage clamp to [-10, 10] | `--custom-advantage-function-path` | [`slime_plugins/kg_rl/advantage_fn.py`](../../slime_plugins/kg_rl/advantage_fn.py) |
| β=0 (no KL penalty) | slime native CLI | `--kl-coef 0` |
| Token-level GRPO loss (broadcast scalar advantage over response tokens) | slime default (`get_grpo_returns`) | — |

## Service topology

```
┌────────────────────────┐  HTTP   ┌─────────────────────────┐
│ SPARQL endpoint        │◄────────│ kg_query_server.py      │
│ (Virtuoso, :8890)      │         │ (Flask, :5501)          │
└────────────────────────┘         │                         │
                                   │  /kg_query              │
┌────────────────────────┐         │  /entity_distance       │
│ Embedding servers      │◄────────│  /health, /metrics      │
│ (vLLM, :8000-:8003)    │         └────────────┬────────────┘
└────────────────────────┘                      │
                                                │ aiohttp
                                   ┌────────────▼────────────┐
                                   │ slime rollout function  │
                                   │ (tree expansion)        │
                                   └────────────┬────────────┘
                                                │ /generate
                                   ┌────────────▼────────────┐
                                   │ SGLang router + workers │
                                   └─────────────────────────┘
```

## End-to-end run

### 0. Start the supporting services

```bash
# (a) SPARQL endpoint — assumes your existing Freebase Virtuoso
# (b) Embedding servers (one example using vLLM for Qwen3-Embedding-8B):
#     vllm serve Qwen/Qwen3-Embedding-8B --port 8000 &
#     vllm serve Qwen/Qwen3-Embedding-8B --port 8001 &
#     vllm serve Qwen/Qwen3-Embedding-8B --port 8002 &
#     vllm serve Qwen/Qwen3-Embedding-8B --port 8003 &

# (c) KG query server (Flask, port 5501)
cd examples/kg_rl
SPARQL_ENDPOINT=http://localhost:8890/sparql \
EMBEDDING_PORTS=8000,8001,8002,8003 \
python kg_query_server.py &
```

### 1. Convert your dataset

```bash
cd examples/kg_rl
python convert_dataset.py \
  --input  /path/to/ISP-KGR/datasets/cwq/train.json \
  --output /root/kg_rl_data/train.jsonl
```

Expected input row schema:
```json
{"question": "...", "answer": ["entity_a", "entity_b"], "mention_entity": [...]}
```

(Field names can differ; edit `convert_dataset.py` if so.)

### 2. Launch training

```bash
export HF_CHECKPOINT=/path/to/Qwen2.5-3B
export REF_LOAD=/path/to/Qwen2.5-3B_torch_dist
export PROMPT_DATA=/root/kg_rl_data/train.jsonl

cd examples/kg_rl
bash run_kg_grpo.sh
```

The launcher starts a Ray head node on `127.0.0.1` and submits the slime
trainer. SGLang workers spin up automatically; training pulls weights into
SGLang after each rollout step via slime's `actor.update_weights()`.

## Knobs (env vars / CLI)

| Variable | Default | Purpose |
| --- | --- | --- |
| `KG_QUERY_URL` | `http://localhost:5501/kg_query` | KG query endpoint |
| `KG_DISTANCE_URL` | `http://localhost:5501/entity_distance` | Entity distance API |
| `KG_BEAM_CANDIDATES` | `16` | K candidates per tree node per round |
| `KG_MAX_INTERACTIONS` | `6` | Tree depth cap |
| `KG_MAX_PROMPT_TOKENS` | `4096` | Force-complete if prompt exceeds this |
| `KG_PER_STEP_MAX_TOKENS` | `1024` | `max_new_tokens` per generation call |
| `KG_MAX_DISTANCE` | `3` | `compute_distance_to_answer` cap |
| `--n-samples-per-prompt` | `6` | GRPO group size after `group_resize` |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.24` | DAPO asymmetric clip |
| `--kl-coef` | `0` | KL penalty (off, matches ISP-KGR) |

## What's NOT yet supported

- Evaluation rollout — `k_beam_tree_rollout` raises `NotImplementedError` when
  `evaluation=True`. Set `--eval-interval -1` or write a single-trajectory
  greedy eval (one for follow-up).
- Partial rollout / aborted-sample recycling — disabled. We do a fresh tree
  per call.
- Multimodal — KG-RL is text-only.

## Provenance of every line of logic

Every transformation in `slime_plugins/kg_rl/` cites the ISP-KGR source line
it was lifted from in its module docstring or inline comments. Run
`grep -rn "ISP-KGR" slime_plugins/kg_rl/` to see them all.

## Troubleshooting

- **`KG error ❌` printed on startup**: kg_query_server isn't reachable. The
  rollout will still run but every query returns `[]`, reward signal collapses.
- **All groups dropped by dynamic filter**: rewards are degenerate (all the
  same). Inspect `metric_gatherer.collect()` output in the rollout log.
- **OOM in SGLang**: lower `--sglang-mem-fraction-static`. Lower
  `KG_BEAM_CANDIDATES`. Lower `KG_PER_STEP_MAX_TOKENS`.
- **`prompt too long`**: lower `KG_MAX_PROMPT_TOKENS` or `KG_MAX_INTERACTIONS`.
