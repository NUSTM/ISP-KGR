#!/bin/bash
# KG-RL: GRPO training over a tree-structured KG rollout.
# Replicates the ISP-KGR training recipe inside slime.
#
# Prerequisites (start in three separate sessions):
#   1) SPARQL endpoint (Virtuoso, port 8890) — your existing Freebase setup
#   2) Embedding servers (vLLM, ports 8000-8003 by default)
#   3) KG query server:
#        cd examples/kg_rl
#        python kg_query_server.py   # listens on :5501
#
# This script then submits the slime training job via Ray.

set -ex

# Stop any orphan SGLang / Ray processes
pkill -9 sglang   || true
sleep 2
ray stop --force  || true
pkill -9 ray python || true
sleep 2

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# Model preset (Qwen2.5-3B — match ISP-KGR's --model_name)
source "${SLIME_ROOT}/scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/root/Qwen2.5-3B/}"
   --ref-load     "${REF_LOAD:-/root/Qwen2.5-3B_torch_dist/}"
   # uncomment to resume / checkpoint
   # --load /root/kg_rl_slime/
   # --save /root/kg_rl_slime/
   # --save-interval 50
)

ROLLOUT_ARGS=(
   # produced by examples/kg_rl/convert_dataset.py
   --prompt-data       "${PROMPT_DATA:-/root/kg_rl_data/train.jsonl}"
   --input-key         messages
   --label-key         answer
   --metadata-key      metadata
   --apply-chat-template
   --rollout-shuffle

   --num-rollout       100
   --rollout-batch-size 128
   --n-samples-per-prompt 6           # GRPO group size after group_resize
   --rollout-max-prompt-len 4096
   --rollout-max-response-len 1024    # per-turn cap (matches per_step_max_tokens)
   --rollout-temperature 1.0
   --rollout-top-p 1.0
   --rollout-top-k 40
   # Stop tokens are set inside the rollout function; this is the engine-wide fallback
   --rollout-stop "</sparql>" "</SPARQL>" "</node>"

   # Custom plug-ins
   --rollout-function-path slime_plugins.kg_rl.rollout_fn:k_beam_tree_rollout
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters:check_reward_nonzero_std

   --global-batch-size 32
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --custom-advantage-function-path slime_plugins.kg_rl.advantage_fn:dapo_grpo_advantage

   # ISP-KGR uses beta=0 (no KL); the asymmetric clip is the DAPO knob.
   --kl-coef 0
   --eps-clip 0.2
   --eps-clip-high 0.24

   # Group-norm (z-score per group). Match ISP-KGR/grpo.py:421.
   --rewards-normalization
   --grpo-std-normalization
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7
   --lr-decay-style constant
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.999
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --bf16
)

# KG endpoints — read by slime_plugins.kg_rl.rollout_fn via env vars
export KG_QUERY_URL="${KG_QUERY_URL:-http://localhost:5501/kg_query}"
export KG_DISTANCE_URL="${KG_DISTANCE_URL:-http://localhost:5501/entity_distance}"
export KG_BEAM_CANDIDATES="${KG_BEAM_CANDIDATES:-16}"
export KG_MAX_INTERACTIONS="${KG_MAX_INTERACTIONS:-6}"
export KG_MAX_PROMPT_TOKENS="${KG_MAX_PROMPT_TOKENS:-4096}"
export KG_PER_STEP_MAX_TOKENS="${KG_PER_STEP_MAX_TOKENS:-1024}"
export KG_MAX_DISTANCE="${KG_MAX_DISTANCE:-3}"

# Ray
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 4 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_ROOT}/Megatron-LM/:${SLIME_ROOT}:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"KG_QUERY_URL\": \"${KG_QUERY_URL}\",
    \"KG_DISTANCE_URL\": \"${KG_DISTANCE_URL}\",
    \"KG_BEAM_CANDIDATES\": \"${KG_BEAM_CANDIDATES}\",
    \"KG_MAX_INTERACTIONS\": \"${KG_MAX_INTERACTIONS}\",
    \"KG_MAX_PROMPT_TOKENS\": \"${KG_MAX_PROMPT_TOKENS}\",
    \"KG_PER_STEP_MAX_TOKENS\": \"${KG_PER_STEP_MAX_TOKENS}\",
    \"KG_MAX_DISTANCE\": \"${KG_MAX_DISTANCE}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 3 \
   --rollout-num-gpus 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
