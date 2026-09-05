# Windows + WSL2 setup (Phase 0)

Flux is developed and benchmarked on a **Windows laptop with CPU only**. The canonical environment is **WSL2 Ubuntu**, not native PowerShell.

## 1. Install WSL2 and Ubuntu

In an Administrator PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot if Windows asks. After Ubuntu launches, create your UNIX user.

Check:

```bash
uname -a          # should mention microsoft-standard-WSL2
python3 --version # 3.11 or 3.12
```

If `python3` is missing: `sudo apt update && sudo apt install -y python3 python3-venv python3-pip build-essential`.

## 2. Docker Desktop with the WSL2 backend

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
2. Settings → General → **Use the WSL 2 based engine**.
3. Settings → Resources → WSL Integration → enable **Ubuntu**.

From Ubuntu, `docker compose version` should work.

## 3. Clone on the Linux filesystem

Do **not** clone under `/mnt/c/...`. Disk I/O there is slow enough to distort CPU benchmarks.

```bash
mkdir -p ~/src
cd ~/src
git clone <this-repo-url> flux
cd flux
```

## 4. CPU PyTorch (not the CUDA wheel)

`make install` creates `.venv` and installs the **CPU** wheel from `https://download.pytorch.org/whl/cpu`. That is the correct index on this laptop. The default PyPI `torch` Linux wheel is CUDA-sized and the wrong default here.

```bash
make install
make hello    # 1000x1000 fp32 tensor on cpu
make test
make api      # naive completions server (downloads Qwen 0.5B on first run)
```

First `make api` downloads ~1 GB of Qwen2.5-0.5B-Instruct into `HF_HOME` (defaults to `./.hf-cache`). Later runs reuse it.

One-command demo after Docker Desktop is wired to Ubuntu:

```bash
docker compose up --build
```

Playground: http://127.0.0.1:3000 — first compose run also downloads those weights into the `hf-cache` volume.

## 5. Line endings

The repo forces LF via `.gitattributes`. If Git on Windows ever checks out CRLF into WSL2, `git add --renormalize .` from Ubuntu.

## 6. What not to do

- Do not install CUDA toolkit, `flash-attn`, or `bitsandbytes` for this project.
- Do not run `uvicorn --workers 4` — that loads four copies of the model.
- Do not treat PowerShell `curl` as real curl; from Windows use `curl.exe -N`, or just curl from WSL2.
