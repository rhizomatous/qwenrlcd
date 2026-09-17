# Running qwenrlcd on one Runpod GPU

The scripts make the environment reproducible and keep large caches on persistent
storage. Pod creation, repository transfer, result retrieval, and stopping billing
remain manual.

## 1. Create the pod

| Setting | Recommendation |
|---|---|
| GPU | 1× A100 80 GB; H100 80 GB is also suitable |
| Template | Current PyTorch/CUDA image with a working NVIDIA driver |
| Persistent volume | At least 50 GB mounted at `/workspace` |
| Repository | `/workspace/qwenrlcd` |

The scripts manage Python 3.12 and project dependencies. The first implementation
uses eager attention with a dense quadratic tree mask, so begin with the tiny smoke
configuration even though the backbone is only 1.7B parameters.

## 2. Put the repository on persistent storage

```bash
cd /workspace
git clone https://github.com/rhizomatous/qwenrlcd
cd qwenrlcd
```

## 3. Initialize

```bash
bash setup/runpod-init.sh
```

This installs `uv` when necessary, installs Python 3.12, syncs training and test
dependencies, verifies CUDA/BF16, runs local tests, and caches
`Qwen/Qwen3-1.7B-Base`. It is idempotent.

 If the Hub rate-limits the pod, run `hf auth login` or export an `HF_TOKEN`.

## 4. Run the smoke test in tmux

```bash
source setup/runpod-activate.sh
tmux new -s qwenrlcd
bash setup/runpod-smoke.sh
```

The preflight uses strict FP32 to test packed/singleton/reordered mask invariance
without BF16 accumulation drift. Training itself remains BF16.

Detach with `Ctrl-b`, then `d`; reattach with `tmux attach -t qwenrlcd`.
