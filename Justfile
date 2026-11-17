# Reproduce the published compound analyses with public inputs.

# S3 bucket and path configuration
S3_BUCKET := "cellpainting-gallery"
S3_PROJECT_PATH := "cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_datastore"
S3_PREFIX := "s3://" + S3_BUCKET + "/" + S3_PROJECT_PATH

# Compute resources (default: all available cores)
CORES := env("CORES", "all")

# Rclone configuration for S3 sync
# Using rclone instead of s5cmd to properly track synced files and avoid re-downloads
RCLONE_FLAGS := env("RCLONE_FLAGS", "--transfers 16 --checkers 16")
RCLONE_BASE := "rclone sync " + RCLONE_FLAGS + " -v --stats-one-line --s3-provider=AWS --s3-region=us-east-1"

# Downloads are anonymous. Bound destructive input sync and preserve generated profiles.
RCLONE_MAX_DELETE := env("RCLONE_MAX_DELETE", "100")
RCLONE_GET := RCLONE_BASE + " --max-delete " + RCLONE_MAX_DELETE
KEEP_GENERATED_PROFILES := "--exclude '*_rsc*.parquet' --exclude '*_with_source7*.parquet' --exclude '*_active_union.parquet'"

# copairs null distributions from one full run. `just reproduce` restores them so copairs
# reads them instead of recomputing them, which was 65% of the job time of a cold run.
# The prefix is a sibling of the datastore, and its README.md
# says how it was made. The files only suit the copairs revision they came from, and the
# sha256 is that of the prefix's MANIFEST.md5, which lists the MD5 of every file. A new
# cache changes the revision, the path and the checksum in one commit.
# NULL_CACHE=0 skips the restore, so copairs computes the nulls itself.
NULL_CACHE := env("NULL_CACHE", "1")
NULL_CACHE_COPAIRS_REV := "897a016"
NULL_CACHE_PATH := "cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_null_cache/copairs-" + NULL_CACHE_COPAIRS_REV + "__run-20261114-jump-repro2"
NULL_CACHE_MANIFEST_SHA256 := "5cfe786e19078f523036f4ed0349279c019d2d6a4871bce3737f90e16db910d4"
NULL_CACHE_SRC := ":s3:" + S3_BUCKET + "/" + NULL_CACHE_PATH

BASELINE_SRC := env("BASELINE_SRC", ":s3:cellpainting-gallery/cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_snapshots/cached-l40s-20261115-04d51ad")
BASELINE_MANIFEST_SHA256 := "f4f4a1473a3717f7695a83610c04039f9ab55ca09145e9a8e3e29597ce7061e8"

# Snakemake configuration
SNAKEMAKE := "pixi run snakemake"

# GPU capacity. Snakemake treats an undeclared resource as UNLIMITED, so without this
# every `gpu=1` rule would be scheduled at once. The count comes from gpu_lease.sh, which
# also hands out the cards -- one source of truth, so the cap can't exceed the cards we
# can actually lease (nvidia-smi ignores CUDA_VISIBLE_DEVICES; gpu_lease.sh honours it).
# Confine a run with:  GPUS=1 just run-all   or   CUDA_VISIBLE_DEVICES=0,1 just run-all
# `|| echo 1` keeps CPU-only hosts schedulable (the rules still run; scripts fall back).
#
# GPUS=0 is NOT the CPU-only knob: it yields `--resources gpu=0`, which Snakemake reads as
# "this rule can never be satisfied", so every gpu=1 rule fails to schedule. To run on CPU
# use CUDA_VISIBLE_DEVICES="" -- the lease then runs jobs unpinned and has_gpu() sends them
# down the CPU path. (Such a run still caps GPU rules at 1 at a time, which costs some
# throughput on a CPU-only box but keeps them schedulable.)
GPUS := env("GPUS", `scripts/gpu_lease.sh --count 2>/dev/null | grep -v '^0$' || echo 1`)
GPU_RESOURCES := "--resources gpu=" + GPUS

# Directory names (following Cookiecutter Data Science structure)
DATA_DIR := "data"
EXTERNAL_DIR := DATA_DIR + "/external"
RAW_DIR := DATA_DIR + "/raw"
INTERIM_DIR := DATA_DIR + "/interim"
PROCESSED_DIR := DATA_DIR + "/processed"

# Default recipe (shows help)
default:
    @just --list

# ==================== MAIN WORKFLOW ====================

# Rebuild everything from a fresh clone and check it. See docs/reproduce.md.
# Jobs at once: one per CPU, up to 32, when the null cache is restored; 32 is the most that
# was measured, and by then the single GPU queue sets the time. Without the cache
# (NULL_CACHE=0) each copairs job fills every CPU by itself, so 4, as in the proven cold run.
# CORES=n overrides both. Memory is the other limit: the rules that need a lot declare
# mem_mb, and Snakemake keeps the sum over running jobs under 90% of this machine's RAM.
# Safe to run again: every step skips work that is already done.
reproduce:
    #!/usr/bin/env bash
    set -euo pipefail
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    ulimit -n 65536
    cores={{ if CORES != "all" { CORES } else if NULL_CACHE == "0" { "4" } else { "$(nproc --all)" } }}  # --all: plain nproc obeys OMP_NUM_THREADS
    [ "{{CORES}}" != all ] || [ "$cores" -le 32 ] || cores=32
    mem_mb=$(awk '/MemTotal/ { print int($2 / 1024 * 0.9) }' /proc/meminfo)
    mkdir -p scratch/rerun
    stage() { echo "$(date -u +%FT%TZ) STAGE $*" | tee -a scratch/rerun/progress.log; }
    # The two downloads run beside the environment install, each in its own process group
    # so that a failure anywhere stops both. Their output goes to scratch/rerun/<name>.log.
    download() { name=$1; shift; setsid "$@" > "scratch/rerun/$name.log" 2>&1 & pids+=("$!"); }
    finish() {
        wait "$2" && return
        tail -20 "scratch/rerun/$1.log" >&2
        echo "reproduce: $1 failed, full log in scratch/rerun/$1.log" >&2
        exit 1
    }
    pids=()
    trap 'code=$?; for pid in "${pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done; stage "exit=$code"' EXIT

    stage setup start
    download inputs sh -c 'just get-inputs && just verify-inputs'
    [ "{{NULL_CACHE}}" = 0 ] || download nullcache just get-null-cache
    pixi install --locked -e default
    for env in test rapids rapids-cheminformatics marimo jump-smiles cheminformatics chemberta deepchem; do
        pixi install --frozen -e "$env"
    done
    # Stop here, in the first minutes, if the pipeline cannot see a GPU.
    scripts/gpu_lease.sh pixi run --frozen -e rapids python -c \
        'import cupy; n = cupy.cuda.runtime.getDeviceCount(); print("GPU devices:", n); assert n > 0'
    pixi run --frozen -e chemberta python -c 'import torch; assert torch.cuda.is_available()'
    stage setup ok

    finish inputs "${pids[0]}"
    stage inputs ok

    if [ "{{NULL_CACHE}}" = 0 ]; then
        stage nullcache skipped
    else
        finish nullcache "${pids[1]}"
        stage nullcache ok
    fi

    snake() { {{SNAKEMAKE}} "$@" {{GPU_RESOURCES}} mem_mb="$mem_mb" --scheduler greedy --consider-ancient download_mitotox=script; }
    targets="all batch_and_source7 explore srijit"
    snake data/interim/image_dimensions.csv --cores 1
    snake $targets --cores "$cores" --printshellcmds --rerun-incomplete --keep-going
    stage pipeline ok

    just validate-figures
    stage figures ok
    left=$(snake $targets --cores "$cores" -n)
    tail -5 <<<"$left"
    grep -q 'Nothing to be done' <<<"$left"
    if [ "{{NULL_CACHE}}" = 0 ]; then
        stage 'baseline skipped: independently computed nulls can differ from the cached baseline'
    else
        just verify-baseline
        stage baseline ok
    fi
    stage done
    echo 'reproduce: ok'

# Check retained or freshly generated outputs against the immutable cached L40S snapshot.
verify-baseline root=".":
    #!/usr/bin/env bash
    set -euo pipefail
    actual=$(cd "{{root}}" && pwd)
    cache="$PWD/scratch/baseline"
    mkdir -p "$cache"
    src="{{BASELINE_SRC}}"
    s3="--s3-provider=AWS --s3-region=us-east-1"
    rclone copyto $s3 "$src/MANIFEST.sha256" "$cache/MANIFEST.sha256"
    echo "{{BASELINE_MANIFEST_SHA256}}  $cache/MANIFEST.sha256" | sha256sum --check --quiet
    # Authenticate the small comparison inputs without downloading the full snapshot.
    : > "$cache/references.sha256"
    for file in CHECKS.sha256 CLAIMS.csv \
        data/processed/commercial-compounds/curated_compounds_{160c_1pc,160c_2pc,320c_1pc,320c_2pc}.csv \
        data/processed/batch-effect/summary.json \
        data/processed/mitotox-analysis/compound_no_source7/summary.json; do
        rclone copyto $s3 "$src/$file" "$cache/$file"
        test -f "$cache/$file"
        awk -v file="$file" 'substr($0, 67) == file' "$cache/MANIFEST.sha256" >> "$cache/references.sha256"
    done
    test "$(wc -l < "$cache/references.sha256")" -eq 8
    (cd "$cache" && sha256sum --check --quiet references.sha256)
    # These authenticated configs differ by checkout root on other machines.
    awk '$2 ~ /^data\/processed\/copairs\/runs\/.*\/results\/\.hydra\/hydra.yaml$/ { print $2 }' "$cache/CHECKS.sha256" > "$cache/hydra-paths.txt"
    rclone copy $s3 --files-from "$cache/hydra-paths.txt" "$src" "$cache"
    # The scientific contract checks the exact retained roster, hashes and claims.
    pixi run python -m jump_production.processing.verify_baseline "$cache" "$actual"
    echo 'baseline: ok'

# Check downloaded inputs against the checksums a full reproduction was run with.
# The release input checksums are immutable.
verify-inputs:
    sha256sum --check --quiet configs/inputs.sha256

# Files of the same name in ~/.copairs are replaced, so later copairs runs on this machine
# use these nulls; other files there stay. A configuration that is not in the cache is
# computed as usual. Safe to run again: a cache that already verifies is not downloaded.
# Restore the copairs null distributions of an earlier full run into ~/.copairs (25 GB).
get-null-cache:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! grep -Eq '^copairs = .*rev = "{{NULL_CACHE_COPAIRS_REV}}"' pyproject.toml; then
        echo "pyproject.toml does not pin copairs {{NULL_CACHE_COPAIRS_REV}}: publish a null cache for the new revision, or run with NULL_CACHE=0" >&2
        exit 1
    fi
    src="{{NULL_CACHE_SRC}}"
    cache="$HOME/.copairs"
    manifest="$PWD/scratch/null_cache/MANIFEST.md5"
    # Anonymous like get-inputs, and `copy`/`copyto` rather than `sync`: nothing local is deleted.
    s3="--s3-provider=AWS --s3-region=us-east-1"
    mkdir -p "$cache" "$(dirname "$manifest")"
    rclone copyto $s3 "$src/MANIFEST.md5" "$manifest"
    echo "{{NULL_CACHE_MANIFEST_SHA256}}  $manifest" | sha256sum --check --quiet
    # This check, not rclone's exit code, proves the restore: rclone exits 0 on an empty listing.
    verify() { (cd "$cache" && md5sum --check --quiet "$manifest" 2>&1 | tail -3); }
    if ! verify >/dev/null; then
        rclone copy $s3 --checksum --fast-list --transfers 64 --checkers 64 \
            --stats 1m --stats-one-line --stats-log-level NOTICE \
            "$src/" "$cache/" --include '/seed*/**'
        verify
    fi
    echo "null cache ok: $(wc -l < "$manifest") files verified in $cache"

# Run core pipeline (db + copairs metrics)
run:
    @echo "Running core pipeline with {{CORES}} cores..."
    @echo "Note: Run 'just get-inputs' first if you haven't downloaded the input data"
    @{{SNAKEMAKE}} core --cores {{CORES}} {{GPU_RESOURCES}} --printshellcmds --scheduler greedy

# Run full pipeline (core + all downstream analysis)
run-all:
    @echo "Running full pipeline with {{CORES}} cores..."
    @{{SNAKEMAKE}} all --cores {{CORES}} {{GPU_RESOURCES}} --printshellcmds --scheduler greedy

# Build all paper figures.
build-figures:
    @echo "Building all paper figures..."
    @{{SNAKEMAKE}} all_figures --cores {{CORES}} {{GPU_RESOURCES}} --printshellcmds --scheduler greedy

# Preview all paper-figure build jobs without writing outputs.
build-figures-dry:
    @{{SNAKEMAKE}} all_figures --cores {{CORES}} {{GPU_RESOURCES}} -n -p --scheduler greedy

# Build one path-safe figure target directly. Rebuilds stale dependencies; never uploads.
build-figure figure_id:
    #!/usr/bin/env bash
    set -euo pipefail
    figure_id={{quote(figure_id)}}
    if [[ ! "$figure_id" =~ ^figure[A-Za-z0-9_]+$ ]]; then
        echo "build-figure: unsafe figure id: $figure_id" >&2
        exit 1
    fi
    echo "Building $figure_id..."
    {{SNAKEMAKE}} "$figure_id" --cores {{CORES}} {{GPU_RESOURCES}} --printshellcmds --scheduler greedy

# Preview one figure build without writing outputs.
build-figure-dry figure_id:
    #!/usr/bin/env bash
    set -euo pipefail
    figure_id={{quote(figure_id)}}
    if [[ ! "$figure_id" =~ ^figure[A-Za-z0-9_]+$ ]]; then
        echo "build-figure-dry: unsafe figure id: $figure_id" >&2
        exit 1
    fi
    {{SNAKEMAKE}} "$figure_id" --cores {{CORES}} {{GPU_RESOURCES}} -n -p --scheduler greedy

# Run exploration notebooks (phase 0 - issue investigations)
explore:
    @echo "Running exploration notebooks with {{CORES}} cores..."
    @{{SNAKEMAKE}} explore --cores {{CORES}} {{GPU_RESOURCES}} --printshellcmds --scheduler greedy

# Preview what will run without executing
dry:
    @echo "Performing dry run..."
    @{{SNAKEMAKE}} --cores {{CORES}} {{GPU_RESOURCES}} -n -p --scheduler greedy
# Run batch correction & source 7 sidebar analysis (#52, #54, #55, #56)
batch-and-source7:
    @echo "Running batch correction & source 7 analysis with {{CORES}} cores..."
    @{{SNAKEMAKE}} batch_and_source7 --cores {{CORES}} {{GPU_RESOURCES}} --printshellcmds --scheduler greedy

# Run Srijit's -seal- analysis notebooks
srijit:
    @echo "Running Srijit's analysis notebooks..."
    @{{SNAKEMAKE}} srijit --cores {{CORES}} {{GPU_RESOURCES}} --printshellcmds

# ==================== DATA SYNC ====================

# Download original upstream sources (these can differ from the pinned reproduction inputs).
get-from-sources:
    @echo "Getting data from original sources..."
    @pixi run python -m jump_production.downloading.download_data

# Download the official ChEMBL archive and the sole historical compatibility
# artifact into a non-mirrored cache. The JUMP compound source is already owned
# by download_data.py under data/external and is verified here without copying it.
get-chembl-target-sources:
    @mkdir -p .cache/chembl-33 .cache/chembl-target-annotations-chembl33/inputs
    @if echo "4e14ace2dc3696cd414c25cbf2370a618fd5ac028713b877725c8ae308e86e8b  .cache/chembl-33/chembl_33_sqlite.tar.gz" | sha256sum --check --status 2>/dev/null; then \
        echo "Verified existing ChEMBL 33 SQLite archive"; \
    else \
        nix shell nixpkgs#aria2 --command aria2c \
            --continue=true \
            --auto-file-renaming=false \
            --allow-overwrite=true \
            --max-connection-per-server=16 \
            --split=16 \
            --min-split-size=16M \
            --file-allocation=none \
            --check-integrity=true \
            --checksum=sha-256=4e14ace2dc3696cd414c25cbf2370a618fd5ac028713b877725c8ae308e86e8b \
            --dir=.cache/chembl-33 \
            --out=chembl_33_sqlite.tar.gz \
            https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_sqlite.tar.gz; \
        echo "4e14ace2dc3696cd414c25cbf2370a618fd5ac028713b877725c8ae308e86e8b  .cache/chembl-33/chembl_33_sqlite.tar.gz" | sha256sum --check --status; \
    fi
    @set -eu; \
        fetch_input() { \
            filename="$1"; \
            source_url="$2"; \
            expected_sha="$3"; \
            destination=".cache/chembl-target-annotations-chembl33/inputs/$filename"; \
            incomplete="$destination.incomplete"; \
            if echo "$expected_sha  $destination" | sha256sum --check --status 2>/dev/null; then \
                echo "Verified existing $filename"; \
                return; \
            fi; \
            curl --fail --location --retry 5 --retry-delay 2 --silent --show-error \
                --output "$incomplete" \
                "$source_url"; \
            if ! echo "$expected_sha  $incomplete" | sha256sum --check --status; then \
                echo "Checksum mismatch for downloaded $filename; refusing to install it" >&2; \
                exit 1; \
            fi; \
            mv "$incomplete" "$destination"; \
            echo "Downloaded and verified $filename"; \
        }; \
        fetch_input \
            jump_to_chembl.csv.gz \
            https://zenodo.org/records/21816611/files/jump_to_chembl.csv.gz \
            50c038187761f3e8bbe998fea22525afe13c3cc01a14178ab3175335464656ac
    @if echo "8885960e92ebd99eb33699a79129f517e668d78dd94f0d7478d39c9825bd3c0a  data/external/compound.csv.gz" | sha256sum --check --status 2>/dev/null; then \
        echo "Verified existing data/external/compound.csv.gz"; \
    else \
        echo "Missing or changed data/external/compound.csv.gz; run 'just get-inputs' or 'just get-from-sources'" >&2; \
        exit 1; \
    fi

# Reproduce and exactly verify the Zenodo ChEMBL 33 matrix outside data/.
build-chembl-targets *ARGS:
    @uv run src/jump_production/downloading/chembl_targets_sqlite.py {{ARGS}}
# Get input data from team S3 (syncs local to match S3 exactly)
get-inputs:
    @echo "Getting input data from team S3..."
    @mkdir -p {{EXTERNAL_DIR}} {{RAW_DIR}}/profiles
    @echo "Syncing external/..."
    {{RCLONE_GET}} ":s3:{{S3_BUCKET}}/{{S3_PROJECT_PATH}}/external/" {{EXTERNAL_DIR}}/
    @echo "Syncing raw/profiles/..."
    {{RCLONE_GET}} {{KEEP_GENERATED_PROFILES}} ":s3:{{S3_BUCKET}}/{{S3_PROJECT_PATH}}/raw/profiles/" {{RAW_DIR}}/profiles/
    @echo "Done!"
# Validate one already-built figure against its standard summary envelope.
validate-figure figure_id:
    @PROCESSED_DIR={{quote(PROCESSED_DIR)}} scripts/validate_figure.sh {{quote(figure_id)}}

# Validate every locally built figure before mirroring data/.
validate-figures:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    # rclone skips a symlink, exits 0, then deletes the objects it stood for, and the
    # post-sync check cannot follow it either, so it agrees. Nothing else here sees this.
    links=$(find {{quote(DATA_DIR)}} -type l)
    [[ -z "$links" ]] || { printf 'validate-figures: symlinks under %s/ publish as deletions:\n%s\n' {{quote(DATA_DIR)}} "$links" >&2; exit 1; }
    # Every directory, not just figure*/ with a summary: a half-built directory that has no
    # summary.json used to be skipped entirely and published unchecked.
    entries=("{{PROCESSED_DIR}}"/figures/*/)
    if [[ ${#entries[@]} -eq 0 ]]; then
        echo "validate-figures: no figure directories found" >&2
        exit 1
    fi
    for entry in "${entries[@]}"; do
        PROCESSED_DIR={{quote(PROCESSED_DIR)}} scripts/validate_figure.sh "$(basename "$entry")"
    done
# ==================== CODE QUALITY ====================

# Run tests (pytest lives in the `test` pixi env; a bare `pixi run pytest` won't find it)
test *ARGS:
    @pixi run -e test pytest {{ARGS}}

# Run snakemake linting (filters out conda/container warnings since we use pixi)
snakelint:
    #!/usr/bin/env bash
    echo "Running snakemake lint..."
    echo "(Note: conda/container warnings are filtered - we use pixi for environment management)"
    echo ""
    output=$({{SNAKEMAKE}} --lint 2>&1)
    # Filter out conda/container-related lines
    filtered=$(echo "$output" | grep -v -E "(Specify a conda environment|https://snakemake.readthedocs.io/en/latest/snakefiles/deployment|This way, the used software|workflow can be executed on any machine|Also see:$)")
    # Only show non-empty lines that aren't just rule headers with no content
    echo "$filtered" | grep -v -E "^Lints for rule.*:$" | grep -v "^[[:space:]]*$" || echo "No lint warnings!"
# See pipeline status
status:
    @echo "Pipeline status:"
    @{{SNAKEMAKE}} --summary

# Release a stale Snakemake lock (left behind when a run is killed or the box dies)
unlock:
    @{{SNAKEMAKE}} --unlock
# ==================== PAPER ====================

# Build, preview, or view the collated dataset-reference report (`build`, `dry`, or `view`)
dataset-reference action="build":
    @case {{quote(action)}} in \
        build) {{SNAKEMAKE}} data/processed/dataset_reference/dataset_reference.json --cores {{CORES}} {{GPU_RESOURCES}} --printshellcmds ;; \
        dry) {{SNAKEMAKE}} -n data/processed/dataset_reference/dataset_reference.json ;; \
        view) ulimit -n 65536; uv run --with marimo marimo run notebooks/dataset_reference.py --sandbox --host 127.0.0.1 ;; \
        *) printf 'Unknown action: %s. Use build, dry, or view.\n' {{quote(action)}}; exit 1 ;; \
    esac

# Check the reviewed source roster and content before committing or archiving.
check-release:
    @pixi run python -m jump_production.release
