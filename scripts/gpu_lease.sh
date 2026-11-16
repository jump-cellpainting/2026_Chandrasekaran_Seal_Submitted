#!/usr/bin/env bash
# Run a command pinned to exactly one free GPU.  (`--count` prints how many are leasable.)
#
# Snakemake's `resources: gpu=1` caps how MANY gpu jobs run at once, but it never assigns
# a device id. Every job would therefore land on device 0: that card OOMs while the others
# sit idle. Here we flock one lock file per leasable GPU and export CUDA_VISIBLE_DEVICES
# for the child, so concurrent jobs spread across the cards and no card is double-booked.
#
# The lock lives on a file descriptor that is inherited across exec, so it is held for
# exactly as long as the job runs and is released by the kernel even if the job is killed.
#
# CUDA_VISIBLE_DEVICES is honoured as the set of cards we may lease from:
#   unset      -> every GPU nvidia-smi reports
#   "0,2"      -> only cards 0 and 2
#   ""         -> CPU-only (the CUDA convention, used by processing/umap.py); run unpinned
#
# `--count` prints the size of that set (0 for CPU-only). The Justfile uses it to set
# `--resources gpu=N`, so Snakemake's cap and the cards we can hand out cannot disagree.
#
# SCOPE: the locks are per-user ($TMPDIR/jump-gpu-locks-$UID), so this arbitrates between
# *your own* concurrent jobs -- which is all it is meant to do. Two users running the
# pipeline on the same box each have their own lock dir, so both can lease card 0 and OOM
# each other. Pointing them at one directory via JUMP_GPU_LOCKDIR does NOT fix that: the
# lock files are opened for writing, and user B cannot write user A's file. Making that
# work needs read-only opens (flock takes LOCK_EX on an O_RDONLY fd) plus mode-0644
# creation -- a real change, not a config tweak. Don't document it as one.
set -uo pipefail

leasable_devices() {
    if [[ -n "${CUDA_VISIBLE_DEVICES+set}" ]]; then
        # Set-but-empty means "no GPUs". Must be distinguished from unset.
        [[ -z "$CUDA_VISIBLE_DEVICES" ]] && return 0
        tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | tr -d ' ' | grep -v '^$'
    else
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '
    fi
}

mapfile -t devices < <(leasable_devices)

if [[ "${1:-}" == "--count" ]]; then
    echo "${#devices[@]}"
    exit 0
fi

if [[ $# -eq 0 ]]; then
    echo "usage: gpu_lease.sh <command> [args...]   |   gpu_lease.sh --count" >&2
    exit 2
fi

# No leasable GPU (CPU-only host, or CUDA_VISIBLE_DEVICES=""): run as-is. The scripts
# fall back to CPU via jump_production.gpu.has_gpu(), and we must not resurrect a GPU
# by exporting one here.
if [[ ${#devices[@]} -eq 0 ]]; then
    exec "$@"
fi

lockdir="${JUMP_GPU_LOCKDIR:-${TMPDIR:-/tmp}/jump-gpu-locks-$(id -u)}"
# Bail rather than run unpinned: a lockdir we cannot create means we cannot arbitrate, and
# silently letting every job pile onto card 0 is the exact OOM this script exists to prevent.
if ! mkdir -p "$lockdir"; then
    echo "gpu_lease.sh: cannot create lock dir '$lockdir' (set JUMP_GPU_LOCKDIR)" >&2
    exit 1
fi

while :; do
    for dev in "${devices[@]}"; do
        # Guard the open: without this, a failure here leaves $fd unset and the next line
        # dies with a bare `fd: unbound variable` instead of saying what actually went wrong.
        if ! exec {fd}>"$lockdir/gpu${dev}.lock"; then
            echo "gpu_lease.sh: cannot open lock file '$lockdir/gpu${dev}.lock'" >&2
            exit 1
        fi
        if flock -n "$fd"; then
            export CUDA_VISIBLE_DEVICES="$dev"
            exec "$@"
        fi
        exec {fd}>&-
    done
    # Every card busy. Only reachable when more gpu jobs are scheduled than there are
    # cards (e.g. a bare `snakemake` with no --resources cap); waiting beats OOMing.
    sleep 5
done
