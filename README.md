# hint_rl (HPRL)

RL training with hint injection for math reasoning, built as flag-gated extensions on top of a
[verl](https://github.com/volcengine/verl) fork. Training runs as multi-node Ray jobs; a hint
*selector* (a served model such as gpt-oss-20b, or an OpenAI-API model) injects hints into rollouts.

Everything this repo needs that lives *outside* the repo (conda, the verl fork, model checkpoints)
is found by **relative position**, not by hardcoded paths. Reproduce the layout below and the
launch scripts work from any location. Full setup walkthrough in [Setup](#setup).

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
(`logs/`) all live *inside* the repo — nothing outside to configure for those.

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

## Setup

The easiest path is the **release bundle** — five files (on our cluster:
`/share5/users/xutao.ma/project/dist/`; verify with `sha256sum -c SHA256SUMS`):

| File | Contents |
|---|---|
| `hint_rl-src-<commit>.tar.gz` | this repo, with full git history |
| `hint_rl-data.tar.gz` | `dataset/`, `custom_data/`, `stephint_data/` (not tracked in git) |
| `verl-hprl-src-<commit>.tar.gz` | the verl fork, with full git history |
| `verl-env.tar.gz` | the complete `verl` conda env ([conda-pack](https://conda.github.io/conda-pack/), relocatable) |
| `claude-memory.tar.gz` | *(optional)* Claude Code project memory — see step 8 |
| `SHA256SUMS` | checksums for the tarballs |

All steps below assume `BASE=/your/base` — any path visible to **all** cluster nodes.

### 1. Unpack the source code

```bash
BASE=/your/base
mkdir -p "$BASE/project" "$BASE/model"

tar -xzf hint_rl-src-*.tar.gz    -C "$BASE/project"   # -> $BASE/project/hint_rl
tar -xzf verl-hprl-src-*.tar.gz  -C "$BASE/project"   # -> $BASE/project/verl (sibling!)
```

(If you got this repo from GitHub instead, `git clone` it to `$BASE/project/hint_rl` and clone the
verl fork next to it — same result.)

### 2. Unpack the data

The training/val parquets are not in git; extract them **into the repo root**:

```bash
tar -xzf hint_rl-data.tar.gz -C "$BASE/project/hint_rl"
# -> $BASE/project/hint_rl/{dataset,custom_data,stephint_data}/
```

Every run script's default `TRAIN_FILE` points into `dataset/`.

### 3. Install conda + the packed env

Install a fresh miniconda at `$BASE/miniconda3` (a fresh install here has correct baked paths, so
the relocation shim below never matters), then unpack the prebuilt `verl` env into it:

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$BASE/miniconda3"

mkdir -p "$BASE/miniconda3/envs/verl"
tar -xzf verl-env.tar.gz -C "$BASE/miniconda3/envs/verl"

source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate verl
conda-unpack        # rewrites the env's baked paths; run ONCE after extracting
```

<details>
<summary>No <code>verl-env.tar.gz</code>? Build the env from scratch</summary>

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda create -n verl python=3.12 -y && conda activate verl
# install the training stack (torch, vllm, ray, transformers, ...) per verl's docs,
# then continue with step 4 (drop the --no-deps flag there so pip pulls verl's deps).
```
</details>

### 4. Install verl into the env (editable)

```bash
pip install -e "$BASE/project/verl" --no-deps --no-build-isolation
```

The packed env deliberately does **not** contain verl — an editable install records an absolute
path, so it must be (re)done on *your* checkout. The flags make it work offline (the env already
has all dependencies). Re-run this any time you move the `verl/` directory.

### 5. Model checkpoints

Place HuggingFace-format checkpoints under `$BASE/model/` using the exact names from the layout
diagram above — you only need the models your runs use: the policy model of your chosen
`run_*.sh`, plus `gpt-oss-20b` if you use the served-selector launchers. (Or point
`MODEL_PATH` / `SELECTOR_MODEL_PATH` anywhere you like.)

### 6. Secrets

```bash
echo 'wandb_key=YOUR_KEY' > "$BASE/project/hint_rl/.envrc"   # git-ignored; read by run scripts
```

### 7. Sanity check

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh" && conda activate verl
python -c "import verl, torch, vllm, ray; print('env OK')"
ls "$BASE/project/hint_rl/dataset" | head -3                  # data in place
ls "$BASE/model"                                              # checkpoints in place
```

### 8. (Optional) Claude Code project memory

The bundle may include `claude-memory.tar.gz`: ~50 notes accumulated while building this project
(bug post-mortems, mechanism explanations, experiment results), indexed by `memory/MEMORY.md`.
If you use [Claude Code](https://claude.com/claude-code), install them as *your* project memory —
Claude Code keys memory to `~/.claude/projects/<slug>/`, where `<slug>` is your hint_rl checkout's
absolute path with every non-alphanumeric character replaced by `-`:

```bash
cd "$BASE/project/hint_rl"
SLUG=$(pwd | sed 's/[^a-zA-Z0-9]/-/g')      # e.g. -home-alice-work-hint-rl
mkdir -p ~/.claude/projects/"$SLUG"
tar -xzf claude-memory.tar.gz -C ~/.claude/projects/"$SLUG"   # -> .../$SLUG/memory/
```

From the next session started in that checkout, Claude loads `memory/MEMORY.md` as its index and
pulls individual notes on demand. The notes are plain markdown, so they also work as
human-readable docs (start at `MEMORY.md`). They reference our cluster's internals — treat them
as team-internal and don't commit or publish them.

Notes:

- **All Python deps must already be in the conda env.** The Ray runtime env does not
  `pip install` anything at job start (our cluster nodes have no internet access).
- **`pip install -e` records an absolute path** — step 4 is per-machine, per-location.

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
shebang (`ray`, `pip`, ...). The recommended setup above avoids the problem entirely: a fresh
miniconda at `$BASE/miniconda3` has correct baked paths, and `conda-unpack` fixes the packed env's.

If you instead **reuse a whole conda tree that was installed at a different path** (copied, or the
same share mounted at a different mount point), those baked paths must still resolve. The scripts
contain a shim for the mount-point case:

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
