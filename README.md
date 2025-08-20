# cifar10-classifier
AI project#1

Here’s a full **README.md** draft you can drop into your repo and commit right away.
It includes: project intro, environment matrix, Mac/Windows setup, troubleshooting, and quick start.

````markdown
# CIFAR-10 Classifier

Reproducible PyTorch training pipeline for CIFAR-10 (ResNet18 baseline).

This project is designed for:
- **Mac (Intel, CPU)** → development & quick testing
- **Windows PC (NVIDIA GPU)** → full GPU training
- Optional `.venv` setup if you don’t want Conda

---

## 🚀 Environment Matrix

| Machine | OS | Python | Torch Stack | How |
|---|---|---|---|---|
| **Mac (Intel, CPU)** | macOS x86_64 | 3.10 | `torch==1.13.1`, `torchvision==0.14.1`, `torchaudio==0.13.1` | **Conda** env (recommended) |
| **Windows PC (NVIDIA GPU)** | Windows | 3.11 (or 3.10) | Latest 2.x + matching CUDA (`cu121`/`cu118`, etc.) | **pip** with CUDA wheel index |
| **Optional (Mac via .venv)** | macOS x86_64 | 3.10 | Same as Mac Intel above | `.venv` + pip |

> ⚠️ Use **one** environment at a time. If you use Conda, don’t also activate `.venv` (and vice-versa).

---

## 🖥️ Mac (Intel) – CPU Conda Environment

```bash
# Install Miniforge (Conda) if needed:
# curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh -o miniforge.sh
# bash miniforge.sh

conda create -n cifar10-py310 python=3.10 -y
conda activate cifar10-py310

# CPU-only, last good Intel-mac build
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch -c conda-forge

# Dev utilities
pip install numpy matplotlib tqdm pytest black isort flake8 pre-commit

# Install/refresh git hooks
pre-commit install

# Verify
python - <<'PY'
import sys, torch, torchvision
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
PY
````

Run training:

```bash
conda activate cifar10-py310
python -m src.train
```

---

## 💻 Windows (NVIDIA GPU) – CUDA Pip Environment

1. Check CUDA version:

```powershell
nvidia-smi
```

Note the **CUDA Version** (e.g. 12.1 or 11.8).

2. Create venv + install:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip

# Pick ONE index-url based on your CUDA version:
# CUDA 12.1 → cu121
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 → cu118
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy matplotlib tqdm pytest black isort flake8 pre-commit
pre-commit install

# Verify
python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Run training (GPU):

```powershell
.\.venv\Scripts\activate
python -m src.train
```

---

## 🔄 Optional: Mac (Intel) via `.venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "torch==1.13.1" "torchvision==0.14.1" "torchaudio==0.13.1"
pip install numpy matplotlib tqdm pytest black isort flake8 pre-commit
pre-commit install
```

---

## 🛠️ Troubleshooting

* **`ModuleNotFoundError: torch`**
  You’re using the wrong interpreter. Check:

  ```bash
  which python
  python -c "import sys; print(sys.executable)"
  python -m pip show torch
  ```

  Paths should point inside your active env (`…/miniforge3/envs/cifar10-py310/...` or `…/.venv/...`).

* **Pre-commit fails after switching envs**
  Reinstall hooks:

  ```bash
  pre-commit install
  ```

* **CUDA mismatch on Windows**
  Pick the right wheel index:

  * CUDA 12.1 → `--index-url https://download.pytorch.org/whl/cu121`
  * CUDA 11.8 → `--index-url https://download.pytorch.org/whl/cu118`

---

## ⚡ Quick Start

**Mac (CPU):**

```bash
conda activate cifar10-py310
python -m src.train
```

**Windows (GPU):**

```powershell
.\.venv\Scripts\activate
python -m src.train
```

---

```

---

Would you like me to also generate a ready-made `conda-env-mac-intel.yml` file alongside this, so you can `conda env create -f conda-env-mac-intel.yml` with one command instead of typing all the installs?
```

