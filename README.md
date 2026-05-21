# ISP-KGR

**Interactive Semantic Parsing with Reinforcement Learning over Knowledge Graphs**

This repository hosts the official training code for the ISP-KGR paper. It is a
fork of [THUDM/slime](https://github.com/THUDM/slime) (an SGLang-native
post-training framework) with a self-contained plug-in implementing our
multi-turn, tree-structured GRPO recipe over a Freebase knowledge graph.

The original demo-style implementation (the version that accompanied the paper
submission) is preserved in the git history; this `main` is the
framework-integrated rewrite that we recommend for any new work building on
top of ISP-KGR.

## What lives where

| Path | Purpose |
| --- | --- |
| [`slime_plugins/kg_rl/`](slime_plugins/kg_rl) | All ISP-KGR-specific Python code: tree rollout, format/progress/outcome reward, advantage clamp, async KG client, group resizing |
| [`examples/kg_rl/`](examples/kg_rl) | Training recipe: launcher script, Flask KG query server, dataset converter, prompt templates, end-to-end README |
| Everything else | Unmodified slime — see [`README_slime.md`](README_slime.md) for upstream docs |

## Quick start

```bash
# 0. Install slime per upstream instructions (see README_slime.md and docker/)
# 1. Convert your ISP-KGR JSON dataset to slime JSONL
cd examples/kg_rl
python convert_dataset.py --input /path/to/train.json --output /root/kg_rl_data/train.jsonl

# 2. Start the KG service (Flask, port 5501) in a separate session
python kg_query_server.py

# 3. Launch GRPO training
bash run_kg_grpo.sh
```

See [`examples/kg_rl/README.md`](examples/kg_rl/README.md) for service topology,
environment knobs, and troubleshooting.

## What's customised vs vanilla GRPO

| ISP-KGR feature | Slime plug-in point | File |
| --- | --- | --- |
| K-beam **tree** rollout with KG environment (one question → branching parents → multiple training groups) | `--rollout-function-path` | [`slime_plugins/kg_rl/rollout_fn.py`](slime_plugins/kg_rl/rollout_fn.py) |
| Per-node `0.1·format + 0.3·progress + 0.6·outcome` reward, max-aggregated across passing trajectories | filled inside rollout before emit | [`slime_plugins/kg_rl/reward_fn.py`](slime_plugins/kg_rl/reward_fn.py) |
| Variable group size after dedup → cyclic-copy / reward-stratified resize to fixed K | called from rollout | [`slime_plugins/kg_rl/group_resize.py`](slime_plugins/kg_rl/group_resize.py) |
| Drop groups whose rewards have zero variance | slime built-in | `--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters:check_reward_nonzero_std` |
| Asymmetric PPO clip (ε_low=0.2, ε_high=0.24) | slime native CLI | `--eps-clip 0.2 --eps-clip-high 0.24` |
| Advantage clamp to [-10, 10] | `--custom-advantage-function-path` | [`slime_plugins/kg_rl/advantage_fn.py`](slime_plugins/kg_rl/advantage_fn.py) |
| β=0 (no KL penalty) | slime native CLI | `--kl-coef 0` |
| Token-level GRPO loss (broadcast scalar advantage over response tokens) | slime default | — |

## Citation

```bibtex
@inproceedings{TODO_ispkgr_paper,
  title  = {Interactive Semantic Parsing with Reinforcement Learning over Knowledge Graphs},
  author = {TODO},
  booktitle = {TODO},
  year   = {TODO},
}
```

## Acknowledgements

- [THUDM/slime](https://github.com/THUDM/slime) — the post-training framework
  this fork is built on.
- [SGLang](https://github.com/sgl-project/sglang) — rollout engine.
- The Freebase KG and the original ISP-KGR research line at NUSTM.

## License

Apache 2.0, inherited from upstream slime. See [`LICENSE`](LICENSE).
