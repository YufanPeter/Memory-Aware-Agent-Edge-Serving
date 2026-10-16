# Getting an AWS GPU instance

Prices are approximate us-east-1 on-demand rates. Check the EC2 console before launching.

## 1. Quota (do this first, approval takes ~1–2 days)
New accounts have **0** vCPU quota for GPU instances.
Console → **Service Quotas** → **AWS services** → **Amazon Elastic Compute Cloud (Amazon EC2)**:
- `Running On-Demand G and VT instances` → request **8** (unit is vCPUs; one `*.xlarge` = 4 vCPUs)
- (optional, cheaper) `All G and VT Spot Instance Requests` → **8**

Suggested justification: *"University course project: benchmarking LLM inference scheduling with llama.cpp on a single small GPU instance."*
Request the quota in the same region you will launch in. Also check whether your course provides AWS Academy / Learner Lab or credits.

## 2. Instance choice
| Type | GPU | vCPU / RAM | ~$/h | Note |
|---|---|---|---|---|
| `g4dn.xlarge` | T4 16 GB | 4 / 16 GB | 0.53 | **recommended**: cheapest, fits an edge-like budget |
| `g6.xlarge` | L4 24 GB | 4 / 16 GB | 0.80 | faster, good value |
| `g5.xlarge` | A10G 24 GB | 4 / 16 GB | 1.01 | |

The memory budget is controlled by `backend.ctx_size` (the KV pool), so a large GPU can still emulate a small one.

## 3. Launch
- AMI: **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** (driver + CUDA preinstalled)
- Storage: 50 GB gp3. Security group: SSH (22) from your IP only.

```bash
ssh -i key.pem ubuntu@<ip>
nvidia-smi
git clone <this repo> && cd Memory-Aware-Agent-Edge-Serving
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
LLAMA_COMMIT=d81235049 scripts/setup_llamacpp.sh   # same llama.cpp commit as local runs
export PATH=$HOME/llama.cpp/build/bin:$PATH
uv tool install huggingface_hub   # provides `hf`
scripts/download_model.sh && scripts/download_hotpotqa.sh
uv run python experiments/run.py configs/experiments/smoke.yaml
```
`monitors.nvidia_smi` turns on automatically when `nvidia-smi` is present, which adds GPU memory to the results.
Use `tmux` for long sweeps. Copy results back with `rsync -av ubuntu@<ip>:~/Memory-Aware-Agent-Edge-Serving/results/ results/`.

## 4. Cost control
- **Stop** the instance when idle. Compute billing stops, but the EBS volume still costs about $4/month for 50 GB. **Terminate** it when you are done.
- Billing → **Budgets**: create a monthly cost budget (e.g. $50) with email alerts.
- Spot is about 60–70% cheaper and fine for resumable sweeps, since finished points are skipped on rerun.
