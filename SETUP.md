# Setup (ELI5)

This page is only about **getting Python ready**. How to write a strategy is in [USAGE.md](USAGE.md).

---

## What you are building

Imagine two kitchens:

| Kitchen | What it is | Use it for |
| --- | --- | --- |
| System Python 3.14 | The laptop’s default Python | **Not** for this engine |
| `fx-server37` | A conda “lab” with Python 3.7 | Engine tests + forex research |

Your **server** runs old science packages (`numpy` 1.21, `pandas` 1.3, `numba` 0.56).  
This lab copies that stack so backtest numbers match the server, and so **Numba** (the fast path) works.

There is **no Docker**. The engine is a folder of Python files. Research scripts import it.

```
~/miniconda3/envs/fx-server37   ← Python lab (shared)
/home/mario/dev/BacktestEngine  ← engine code (this repo)
/home/mario/dev/forex-research  ← strategies / studies (imports the engine)
```

---

## One-time: Miniconda + lab (already done on this laptop)

If `conda` already works and `fx-server37` exists, skip to [Every day](#every-day-before-you-run-anything).

### 1. Install Miniconda (once)

```bash
# Download installer (Linux x86_64), then:
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda --version
```

Accept Anaconda channel terms if conda asks (newer conda versions require this once).

### 2. Create the Python 3.7 lab

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n fx-server37 python=3.7 -y
conda activate fx-server37
```

### 3. Install the science core (conda binaries — important for Numba speed)

```bash
conda install -n fx-server37 -c conda-forge \
  "numpy=1.21.6" "pandas=1.3.5" "numba=0.56.4" "llvmlite=0.39.1" \
  "scipy=1.7.3" "scikit-learn=1.0.2" "statsmodels=0.13.5" -y
```

### 4. Install the rest from the cleaned pin files

```bash
conda activate fx-server37
pip install -r /home/mario/dev/BacktestEngine/requirements-server.txt
pip install -r /home/mario/dev/forex-research/requirements-server.txt
```

Those files come from the server dump  
`/home/mario/external_storage/temp_transfer/vscode/requirements.txt`, cleaned:

- broken `certifi @ file:///croot/...` → normal `certifi`
- `forexconnect` **omitted** (live FXCM only; not needed for backtests)

### 5. Optional: Jupyter kernel

```bash
conda activate fx-server37
python -m ipykernel install --user --name fx-server37 --display-name "Python 3.7 (fx-server37)"
```

In notebooks, pick kernel **Python 3.7 (fx-server37)**.

---

## Every day (before you run anything)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fx-server37
```

Your prompt should show `(fx-server37)`. Then:

```bash
python -V
# expect: Python 3.7.x
```

**Do not** use:

- `/usr/bin/python3` (3.14 on this machine)
- `/home/mario/dev/forex-research/.venv` (also 3.14)

---

## Check the lab is healthy (2 minutes)

```bash
conda activate fx-server37
cd /home/mario/dev/BacktestEngine

python -c "import numpy,pandas,numba; print(numpy.__version__, pandas.__version__, numba.__version__)"
# expect: 1.21.6 1.3.5 0.56.4

python -c "from engine_jit import _HAS_NUMBA; assert _HAS_NUMBA; print('Numba OK')"
# if this fails, fix Numba before any strategy work — otherwise backtests are slow

python -m pytest
# expect: all tests passed
```

---

## How research finds the engine

The engine lives at `/home/mario/dev/BacktestEngine` (next to `forex-research`, not inside it).

**Option A — env vars (works from any folder):**

```bash
export BACKTEST_ENGINE_ROOT=/home/mario/dev/BacktestEngine
export PYTHONPATH="$BACKTEST_ENGINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

**Option B — run from the engine folder:**

```bash
cd /home/mario/dev/BacktestEngine
python your_script.py
```

**Option C — path helper in research** (`forex-research/Example/engine_paths.py`):

- reads `BACKTEST_ENGINE_ROOT` first
- else uses `/home/mario/dev/BacktestEngine`
- then calls `ensure_engine_on_sys_path()`

Smoke test from research:

```bash
conda activate fx-server37
cd /home/mario/dev/forex-research
PYTHONPATH=/home/mario/dev/BacktestEngine python -c "from engine import OOEngine; print(OOEngine)"
```

---

## Tiny glossary

| Word | Meaning |
| --- | --- |
| **conda** | Tool that creates sealed Python “labs” |
| **Miniconda** | Small install of conda (not the huge Anaconda bundle) |
| **env / lab** | One named box, e.g. `fx-server37` |
| **pin** | Exact package version (so laptop ≈ server) |
| **Numba** | Makes the fill loop fast; must import successfully |
| **PYTHONPATH** | Extra folders where Python looks for `import engine` |

---

## When something breaks

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No module named engine` | Wrong folder / no PYTHONPATH | `cd` to engine or set `BACKTEST_ENGINE_ROOT` |
| `No module named numpy` / wrong versions | Wrong Python | `conda activate fx-server37` |
| `_HAS_NUMBA` is False | Numba missing or broken | Reinstall numba via conda-forge (step 3) |
| Tests fail after OS upgrade | Env corrupted | Recreate `fx-server37` from this page |

Next step after setup: [USAGE.md](USAGE.md) (load data → write strategy → `OOEngine.run()`).
