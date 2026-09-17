# Setting up qwenrlcd

## 1. Create the pod

Recommended starting point:

| Setting | Value |
|---|---|
| GPU | 1× A100 80 GB (H100 80 GB is also suitable) |
| Template | A current PyTorch/CUDA image with a working NVIDIA driver |
| Persistent volume | At least 50 GB, mounted at `/workspace` |
| Repository location | Somewhere below `/workspace` |

The scripts install their own Python 3.12 environment. They only depend on the
image for the NVIDIA driver, basic shell tools, and `curl`.

## 2. Put the repository on persistent storage

Clone it:

```bash
cd /workspace
git clone rhizomatous/qwenrlcd
cd qwenrlcd
```

## 3. Initialize the environment

```bash
bash setup/runpod-init.sh
```

If Hugging Face ratelimits the pod, authenticate before initialization with `hf auth login` or export an `HF_TOKEN`.

## 4. Set up to run

Use `tmux` so an SSH disconnect does not kill training:

```bash
source setup/runpod-activate.sh
tmux new -s qwenrlcd
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t qwenrlcd
```
