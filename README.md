# hint_rl (HPRL)

RL training with hint injection for math reasoning, built as flag-gated extensions on top of a
[verl](https://github.com/volcengine/verl) fork. Training runs as multi-node Ray jobs; a hint
*selector* (a served model such as gpt-oss-20b, or an OpenAI-API model) injects hints into rollouts.

This README documents the **directory-layout contract**: everything this repo needs that lives
*outside* the repo (conda, the verl fork, model checkpoints) is found by **relative position**, not
by hardcoded paths. Reproduce the layout below and the launch scripts work from any location.

## Directory layout (the contract)

```
<BASE>/                          # any path, on a filesystem all nodes can see
├── miniconda3/                  # conda install; must contain the `verl` env
├── model/                       # HuggingFace-format checkpoints, by exact name
│   ├── gpt-oss-20b/                 # selector model (launch_hprl_cluster.sh mode)
│   ├── Qwen3-8B-Base/
│   ├── Qwen3-4B-Instruct-2507/
│   ├── Qwen2.5-7B-Instruct/
│   ├── Olmo-3-7B-Instruct-SFT/
│   ├── Olmo-3-7B-Instruct-DPO/
│   └── Olmo-3-1025-7B/
└── <project>/                   # any name (ours is `project/`)
    ├── hint_rl/                 # this repo
    └── verl/                    # the verl fork — MUST be a sibling of hint_rl
```

Only the *relative* positions matter, and only these three:

1. `verl/` is a **sibling** of `hint_rl/` (`hint_rl/../verl`).
2. `miniconda3/` is **two levels above** the repo (`hint_rl/../../miniconda3`).
3. `model/` is **two levels above** the repo (`hint_rl/../../model`).

Datasets (`dataset/*.parquet`), reward functions (`reward/`), checkpoints (`ckpt/`), and logs
(`logs/`) all live *inside* the repo — nothing to set up for those.

## How the scripts find things

Every launch/run script self-locates from its own file path and derives all external paths from
there. The preamble is identical everywhere (see e.g. `script/ray_cluster_launch.sh`):

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HINT_RL_HOME=${HINT_RL_HOME:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}    # repo root
PROJECT_HOME=${PROJECT_HOME:-"$(cd "${HINT_RL_HOME}/.." && pwd)"}  # holds hint_rl + verl
BASE_HOME=${BASE_HOME:-"$(cd "${PROJECT_HOME}/.." && pwd)"}        # holds miniconda3 + model
CONDA_HOME=${CONDA_HOME:-"${BASE_HOME}/miniconda3"}
VERL_HOME=${VERL_HOME:-"${PROJECT_HOME}/verl"}
```

(Scripts nested deeper, e.g. `script/hint_rl/*.sh`, use `../..` — they all resolve to the same
repo root. Keep this in mind if you add a launcher in a new subdirectory.)

Because every assignment is `${VAR:-default}`, you can override any of them per-run via the
environment instead of editing scripts. The main knobs:

| Variable | Default | Meaning |
|---|---|---|
| `HINT_RL_HOME` | auto (from script location) | this repo's root |
| `VERL_HOME` | `<repo>/../verl` | verl checkout; becomes the Ray job's `--working-dir` |
| `CONDA_HOME` | `<repo>/../../miniconda3` | conda install |
| `CONDA_ENV` | `verl` | conda env name to activate |
| `MODEL_PATH` | `${BASE_HOME}/model/<name>` (per run script) | policy checkpoint |
| `SELECTOR_MODEL_PATH` | `${BASE_HOME}/model/gpt-oss-20b` | selector model (served-selector mode) |
| `TRAIN_FILE` | `<repo>/dataset/<name>.parquet` (per run script) | training data |
| `CONDA_INSTALL_PREFIX` | site-specific, see below | conda relocation shim |

A few run scripts pin `MODEL_PATH` internally (e.g. the Qwen3-base and Olmo scripts build a
`<name>-hprl` wrapper dir under `${BASE_HOME}/model/` with fixed eos/chat-template config); they
still only need the *source* checkpoint present under `model/`.

## Setting up from a fresh clone

```bash
BASE=/path/of/your/choice
mkdir -p "$BASE/project" "$BASE/model"

git clone <this-repo>  "$BASE/project/hint_rl"
git clone <verl-fork>  "$BASE/project/verl"

# 1. conda: install INTO $BASE/miniconda3 (a fresh install here needs no relocation shim)
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$BASE/miniconda3"
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda create -n verl python=3.12 -y && conda activate verl
# ...install training deps (torch, vllm, ray, ...) into this env, then:
pip install -e "$BASE/project/verl"

# 2. models: place HF checkpoints under $BASE/model/ using the names listed above
#    (or export MODEL_PATH / SELECTOR_MODEL_PATH to point elsewhere)

# 3. secrets (git-ignored): wandb key read by the run scripts
echo 'wandb_key=YOUR_KEY' > "$BASE/project/hint_rl/.envrc"
```

Notes:

- **`pip install -e` records an absolute path.** If you later move `verl/`, re-run
  `pip install -e` (training itself still works via Ray's `--working-dir`, but any direct
  `import verl` from the env — e.g. the eval/merge tooling — resolves through the stale path).
- **All Python deps must already be in the conda env.** The Ray runtime env does not
  `pip install` anything at job start (our cluster nodes have no internet access).

## Launching

Entry points (each script's header comments document its required/optional env):

- `script/ray_cluster_launch*.sh` — plain multi-node run: brings up Ray across all pods, head runs
  the `TRAIN_SCRIPT` (a sibling `run_*.sh`, which `ray job submit`s the verl job).
- `script/hint_rl/launch_hprl_cluster*.sh` — HPRL runs: splits pods into training nodes + selector
  nodes serving `gpt-oss-20b` (the `*_openai*` variants use an API selector instead, no selector pods).
- `script/stephint_baseline/ray_cluster_launch_*.sh` — solution-prefix baseline.

Run the chosen launcher **on every pod** of a multi-pod job (e.g. a PyTorchJob), with
`MASTER_ADDR`/`MASTER_PORT` (and a per-pod `RANK` for the cluster-split launchers) injected by the
platform. Point the job's entrypoint at the launcher's **absolute path** — that path lives in your
job spec, so remember to update it if you relocate the tree.

## Conda is not relocatable (`CONDA_INSTALL_PREFIX`)

Conda bakes its absolute install prefix into `etc/profile.d/conda.sh` and into every entry-point
shebang (`ray`, `pip`, ...). If you install conda fresh at `$BASE/miniconda3` you can ignore this
section. If you instead **reuse a conda tree that was installed at a different path** (copied, or
the same share mounted at a different mount point), those baked paths must still resolve. The
scripts contain a shim for the mount-point case:

```bash
CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX:-<path that contained miniconda3 at install time>}
if [ ! -e "${CONDA_INSTALL_PREFIX}" ]; then
    sudo ln -sfn "${BASE_HOME}" "${CONDA_INSTALL_PREFIX}"   # baked paths resolve again
fi
```

Set `CONDA_INSTALL_PREFIX` to the directory that *contained* `miniconda3/` when it was installed.
The shim only fires when that path does not exist on the current machine; if it exists but holds a
*different* (or deleted) conda, the shim cannot help — create the symlink yourself or do a fresh
install.

## Relocating an existing tree

Moving `hint_rl/` + `verl/` (keeping them siblings, with `miniconda3/` and `model/` two levels up)
is supported — the scripts carry no hardcoded repo paths. Checklist:

1. Preserve the layout geometry above.
2. Re-run `pip install -e <new>/verl` in the conda env.
3. Conda: either leave `miniconda3/` at its original path (nothing else needed — the scripts will
   still find it if the geometry holds), or handle the baked prefix per the section above.
4. Update the absolute launcher path in your cluster job spec.
5. If you copy instead of `mv`, include dotfiles (`.envrc`).
