# MiMo-V2.6-Flash-MOPD EXL3 — one-command Podman deploy (AMD Strix Halo 128 GB)

Serves [yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz](https://huggingface.co/yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz)
(Kyojin engine built on ExLlamaV3, EXL3 quantisation, gfx1151, DFlash speculative decoding) in a
rootless Podman container on an AMD Strix Halo (Ryzen AI Max+ 395).

One command brings it up; one command tears it down.

```bash
cp .env.sample .env   # optional — every var has a default
./start.sh            # build (cached) -> download model -> run -> wait healthy
./stop.sh
```

## What's here

| File            | Role                                                                                                                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Containerfile` | Self-contained runtime: uv Python 3.12, pinned ROCm 7.0+ nightly wheels (torch/rocm-sdk-core/rocm-sdk-devel for gfx1151), Kyojin engine cloned from GitHub and compiled. Checksum-pinned git revision. |
| `start.sh`      | One-command entrypoint: preflight → build (cached) → download checkpoint (resumable, SHA256SUMS-verified) → run → health-poll.                                                                  |
| `stop.sh`       | Stops the container with a 60 s grace period; idempotent.                                                                                                                                       |
| `.env.sample`   | Optional overrides — copy to `.env`.                                                                                                                                                            |

The ~107 GB checkpoint is **not** baked into the image; it is bind-mounted read-only at `/models`. The image is built on first run and cached for subsequent launches.

## Requirements

- **Podman** (rootless) — `podman --version` ≥ 5.x
- **AMD GPU driver** present: `/dev/kfd` and `/dev/dri/renderD128` exist
- Current user in the **`render`** (and ideally `video`) group — check with `id -nG`
- **~140 GB free disk**: ~107 GB weights + ~30 GB image + caches
- **128 GB unified RAM** (the default `-c 262144` (256K context) leaves ~21 GB free with the vision tower loaded)
- `curl`, `python3`, and `hf` (from `pip install huggingface_hub[cli]`; older releases call it `huggingface-cli`) or the `huggingface_hub` Python package on the host for the first-model download

No root/sudo is required for the normal path. Podman runs rootless; GPU access comes from `--group-add keep-groups` (carries your host `render` group into the container) plus SELinux `:z` bind relabeling.

## Quick start

```bash
cp .env.sample .env      # optional
./start.sh
```

First run:

1. Clones and builds the Kyojin engine with ROCm gfx1151 support (~10–20 min).
2. Downloads the ~107 GB checkpoint to `./models` (resumable via `hf` or `huggingface_hub`), then verifies every shard against the pack's `SHA256SUMS`.
3. Starts the container and streams the engine log to `.mimo26.log`.
4. Exits "ready" once `/health` returns 200.

Then:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[0].id'
# -> yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz
```

Tear down:

```bash
./stop.sh
```

Re-running `./start.sh` after the first time skips the build and the download and reaches ready materially faster (Kyojin kernel/GEMM autotune caches live in `./.cache`).

## Configuration

Copy `.env.sample` to `.env` and edit. Precedence (highest first): **exported shell env → `.env` → `start.sh` defaults.**

| Var              | Default                                  | Meaning                                                             |
| ---------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| `IMAGE`          | `mimo26-flash-strix:latest`              | Image tag                                                           |
| `CONTAINER_NAME` | `mimo26-flash-strix`                     | Container name                                                      |
| `PORT`           | `8080`                                   | Host publish port (container always listens on 8000)                |
| `MODEL_DIR`      | `./models`                               | Where the checkpoint lives / downloads to                           |
| `CACHE_DIR`      | `./.cache`                               | Kyojin/HIP/HF caches (persist kernel/GEMM autotune across restarts) |
| `HF_MODEL`       | `yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz` | HuggingFace model repo                                            |
| `HF_REVISION`    | *(latest)*                               | Pin a commit SHA for reproducibility                                |
| `LOG_FILE`       | `./.mimo26.log`                          | Engine output log                                                   |
| `READY_TIMEOUT`  | `3600`                                   | Seconds to wait for the health endpoint                             |
| `MAX_JOBS`       | `$(nproc)`                               | Kernel build parallelism (hipcc/ninja); also `podman build --jobs`  |
| `NO_VISION`      | *(unset)*                                | Any non-empty value passes `--no-vision` (saves 1.4 GiB GPU memory; image requests get HTTP 400) |
| `CTX_SIZE`       | `262144`                                 | Context window in tokens (256K needs ~105 GiB unified memory; lower it when other apps use RAM) |
| `HF_TOKEN`       | *(unset)*                                | Optional HuggingFace token — higher download rate limits            |

## How it works

The `Containerfile` builds a self-contained image:

1. Ubuntu 24.04 base with system build deps.
2. `uv` for Python management.
3. Kyojin engine git-cloned from GitHub at a pinned commit.
4. ROCm gfx1151 PyTorch + `rocm-sdk-core` + `rocm-sdk-devel` wheels installed into a uv-managed venv (self-contained, no system ROCm), then `rocm-sdk init` expands the devel headers.
5. `build.sh` compiles the exllamav3 C++/HIP extension for gfx1151.
6. At runtime, `env.sh` sets `LD_PRELOAD` (HSA runtime) and `PYTHONPATH`, then `serve.py` serves the model on `0.0.0.0:8000` (host port mapped by `start.sh`) under the HuggingFace repo id, with `-c 262144` (256K context, overridable with `CTX_SIZE`). The DFlash drafter is auto-found in `<model>/drafter/` (speculation on by default, `--ndt 7 --dynamic-draft --draft-confidence 0.6`); the vision tower loads from `<model>/vision/vision.safetensors` unless `NO_VISION` is set. `$HOME/.cache` is a symlink to the `/cache` bind-mount so GEMM tuning and kernel caches survive container recreation.

The image does **not** contain a system ROCm — the wheels are self-contained and the `_rocm_sdk_devel` package provides both the ROCm headers for compilation and the compatible HSA runtime for execution.

## Recommended sampling

From the model card (upstream values):

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz",
    "messages": [{"role": "user", "content": "Explain what a KV cache is, briefly."}],
    "max_tokens": 256
  }' | jq -r '.choices[0].message'
```

The server defaults to the pack's `generation_config.json` (`temperature 1.0`, `top_p 0.95`) when a request omits or nulls the field. Greedy decoding is not recommended, especially for long outputs.

For agent use with reasoning:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz",
    "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
    "max_tokens": 512,
    "reasoning_effort": "high"
  }'
```

## Image input

The server loads the vision tower from `<model>/vision/vision.safetensors` at start (1.4 GiB GPU memory, about 2 s). An image goes in a user message as an OpenAI `image_url` part; the URL is a `data:` URI or a local file path (the server does not fetch remote URLs). One image takes 24 to about 1,100 tokens depending on its size (capped at 2,097,152 pixels). Audio and video are not supported.

To skip the vision tower (saves 1.4 GiB GPU memory; image requests then get HTTP 400):

```bash
NO_VISION=1 ./start.sh
```

## Troubleshooting

**First launch is slow (5–15 min)** — the engine compiles dense GEMM kernels on the first requests. The progress is printed to the log every 30 seconds. This only happens on a cold cache; subsequent launches warm up fast.

**`Cannot allocate memory` / HSA pin errors at startup** — the user session's `memlock` limit is too low for ROCm memory pinning. `start.sh` only warns about this (it does not block). If the engine actually fails to pin, raise it once (needs sudo):

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee /etc/systemd/system/user@.service.d/10-memlock.conf
sudo systemctl daemon-reload
# then log out and back in
```

**SELinux `Permission denied` on `/models`** — confirm the `:z` suffix on the bind (start.sh sets it). If you relabeled the directory manually, `restorecon -R` it or just re-run `./start.sh`.

**Host OOM under load** — close other apps or lower the context window with `CTX_SIZE` (for example `CTX_SIZE=32768 ./start.sh`, or set it in `.env`)

**`hf: command not found`** — install `huggingface_hub` on the host:

```bash
pip install huggingface_hub[cli]
```

Or ensure `python3`'s environment has `huggingface_hub` installed. start.sh tries `hf` first, then a Python one-liner using `huggingface_hub`.

**SHA256SUMS verification failed** — the pack ships `SHA256SUMS`; start.sh verifies every shard after download. If it fails, re-run `./start.sh` to re-download the corrupted files.

**Container exits immediately** — check logs:

```bash
podman logs mimo26-flash-strix
```

Common causes: missing GPU driver (`/dev/kfd` not present), `memlock` too low, or the model directory not containing `config.json`.

**No `/dev/kfd` or `/dev/dri`** — the AMD GPU kernel driver is not loaded. Load it:

```bash
sudo modprobe amdgpu
sudo modprobe kfd
```

## Notes

- **No system ROCm in the image.** The wheels are self-contained (`_rocm_sdk_devel` inside the venv); the vendor docs forbid mixing a system ROCm. The container distro (Ubuntu 24.04) is independent of the host distro — only the kernel is shared.
- **`--ipc host` is load-bearing** — the GPU runtime dies at startup without it; no `shm_size` substitutes.
- **Docker instead of Podman** — the same `Containerfile` works under a rootful Docker daemon with `docker run --device /dev/kfd --device /dev/dri --group-add video,render --ipc host --ulimit memlock=-1:-1`. Swap the CLI in `start.sh` if you prefer Docker.
- **DFlash drafter** — the 4 bpw drafter in `drafter/` is auto-found next to the shards; speculation is on by default (`--ndt 7 --dynamic-draft --draft-confidence 0.6`). Delete the folder, or set `MIMO_SPEC=0`, for plain decode. If the drafter folder is missing, the server logs one line and decodes plain.
- **Tuning overlay** — `zz-e2e-step120.safetensors` (2.3 MB) is a small overlay from one end-to-end tuning stage; it replaces the scale vectors and norm weights of layers 24 to 47. Our engine loads every `*.safetensors` file in the folder and lets the later file win. If you delete it you get the pack without the tuning stage (KLD 1.9 % higher on the same rows).

## Model speed reference (AMD Strix Halo, Ryzen AI Max+ 395, 128 GB)

| Metric                                  | Value       |
| --------------------------------------- | ----------- |
| Prefill (4K prompt, with drafter)       | ~650 tok/s  |
| Decode (plain, no drafter)              | 28.9 tok/s  |
| Decode (speculative, prose)             | 32.1 tok/s  |
| Decode (speculative, chat)              | 34.8 tok/s  |
| Decode (speculative, code)              | 40.8 tok/s  |

Speed figures from the [model card](https://huggingface.co/yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz) (medians of 6 runs over two server loads, 128 new tokens, client temperature 0, `-c 32768`).

## Licence

The model weights are MIT-licensed (see the Hugging Face model card). The Kyojin engine is MIT-licensed. See the upstream repos for full licence details.

- Base model: [XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) — Copyright 2026 Xiaomi Corporation (Apache-2.0 for `modeling_mimo_v2.py` / `configuration_mimo_v2.py`)
- Quantisation: [yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz](https://huggingface.co/yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz) — Copyright (c) 2026 Yamz Labs, MIT
- Engine: [Yamz-Labs/kyojin](https://github.com/Yamz-Labs/kyojin) — MIT
- This recipe: authored and published by [piotreknow02](https://github.com/piotreknow02/MiMo-V2.6-Flash-Strix-Halo) — provided as-is, without warranty
