# Reproduce on AWS

This guide takes a new user or agent with their own AWS account from a fresh checkout to complete results and a verified outcome.
A Mac or Linux computer launches one Linux GPU instance, and the pipeline runs on that instance.
The local computer needs no GPU, Nix, pixi or direnv: the instance installs Nix, and the repository's pinned environment supplies the rest.
Inputs, the null cache and the comparison baseline are public downloads, so no S3 write access or other credentials are needed.
Without AWS, use the [Slurm route](#optional-slurm-without-nix-or-root).

A run has two separate outcomes.
The workflow completes when every pipeline job and every figure validation succeeds.
The results agree with the baseline when they also match a pinned snapshot that was computed on an NVIDIA L40S.
On the L40S instance used below, expect both.
On [other GPU hardware](#outcomes-on-other-gpu-hardware) the workflow can complete while some results differ, so report the two outcomes separately.

## 1. Prepare the computer that launches AWS

Use a terminal on macOS or Linux with Bash, Git, curl, OpenSSH (`ssh`, `ssh-keygen`), rsync and AWS CLI v2.
On macOS, `xcode-select --install` provides Git if it is missing.
On Ubuntu or Debian, install the shell tools with:

```bash
sudo apt-get update
sudo apt-get install -y git curl openssh-client rsync ca-certificates
```

Install AWS CLI v2 with [AWS's macOS or Linux installer](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), then check the required commands:

```bash
for tool in bash git curl ssh ssh-keygen rsync aws; do
    command -v "$tool" || break
done
aws --version
mkdir -p ~/.ssh
```

Install any missing command before continuing.
The computer needs outbound HTTPS, outbound SSH (port 22) to the instance, and several GB free: the pulled results are about 1.6 GB.

## 2. Prepare your AWS account

The run creates billable GPU compute and disk in your account.
If you have no account or CLI profile yet, follow AWS's [account and CLI prerequisites](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-prereqs.html) and [authentication guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-authentication.html). Set the profile and region and confirm the account; use your own profile name if one is already configured, and skip `aws login` when its session is valid:

```bash
export AWS_PROFILE=jump-reproduction
export AWS_REGION=us-west-2
aws login --profile "$AWS_PROFILE" --region "$AWS_REGION"
aws sts get-caller-identity
```

`aws login` needs AWS CLI 2.32.0 or later and the permission described in [AWS's login instructions](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html); it opens a browser, so add `--remote` on a computer without one.
With IAM Identity Center, use `aws sso login --profile "$AWS_PROFILE"` instead.
Check that `get-caller-identity` names the account you intend to use, and renew an expired session before running launcher commands.
Temporary environment credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`) also work if they outlive each command; a command that is already waiting keeps its original credentials, and termination alone can wait about six minutes.

The account needs:

- EC2 permission to inspect images, instances, subnets, key pairs and security groups; launch and tag instances; import and delete key pairs; create and delete security groups and authorize their SSH ingress; and terminate instances.
  The recovery commands in step 5 also need permission to start instances and modify their instance type.
- At least 32 available vCPUs in the **Running On-Demand G and VT instances** quota in `us-west-2`, for the single-L40S `g6e.8xlarge` used below.
  New accounts may have zero; [request a quota increase](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html) before launching.
  Quota and available GPU capacity are separate requirements.
- A [default VPC with default public subnets](https://docs.aws.amazon.com/vpc/latest/userguide/default-vpc.html) in `us-west-2`, with public IP assignment and internet access.
  The launcher uses these subnets; it does not create a VPC or adapt to a private-only network.

If any of these is missing, resolve it with the account administrator before proceeding.
Do not switch to an untested GPU or region to get around it: the baseline uses L40S hardware and the image is specific to the region.

## 3. Clone and launch

Run every command of this AWS route directly in the local terminal, without a Nix shell or `direnv`.
The launcher finds your instance by `NAME`.
Keep the same `AWS_PROFILE`, `AWS_REGION`, `NAME`, `AMI` and `TYPES` exported in every shell that runs it, until cleanup is finished; save their values if you change terminals.
Use a new `NAME` for each run so that it cannot select another instance.

The public repository clones without GitHub credentials.
The example selects `main`; to reproduce a particular revision, such as the one this copy of the guide belongs to, set `RELEASE_REF` to its full commit ID instead.
Record the ID that `git rev-parse HEAD` prints with your results.

```bash
export REPOSITORY_URL=https://github.com/jump-cellpainting/2026_Chandrasekaran_Seal_Submitted.git
export RELEASE_REF=main
git clone "$REPOSITORY_URL" jump-reproduction
cd jump-reproduction
git checkout "$RELEASE_REF"
git switch -c reproduction
git rev-parse HEAD

export NAME="jump-repro-$(date -u +%Y%m%d-%H%M%S)"
export AMI=ami-0b87420e2a817294d
export TYPES=g6e.8xlarge
export RESULTS="$PWD/scratch/$NAME"
printf 'profile=%s region=%s name=%s results=%s\n' "$AWS_PROFILE" "$AWS_REGION" "$NAME" "$RESULTS"

scripts/ec2.sh up
scripts/ec2.sh push
```

`up` creates one L40S instance with 32 CPUs, 248 GB RAM and a 250 GB disk, plus its own key pair and a security group that admits SSH only from your current public IP.
`push` sends the committed `reproduction` branch as a Git bundle, so the instance needs no GitHub credentials; uncommitted edits are not sent.
Keep the `AMI` pin for baseline comparisons: that Ubuntu image carries NVIDIA driver `595.112.07`.
`TYPES=g6e.8xlarge` keeps the launch within the 32-vCPU quota.

If `up` ends with `No capacity for any listed type in any zone.`, run it again with the same name, now or later; it reuses the key pair and security group it already created.
With enough quota you can instead set `TYPES` to another tested L40S type: `g6e.12xlarge` needs 48 vCPUs and `g6e.16xlarge` needs 64.

If `up` times out waiting for SSH (five minutes by default, `SSH_WAIT_SECONDS`), the instance is still running and billed.
Check the reported SSH error, your key and whether your public IP still matches the security group, then continue with `scripts/ec2.sh push`.
Do not rerun `up` in that case; it reports "already exists".

If you abandon a launch at any point, run the cleanup command in step 5 with the same name, because a key pair, a security group or an instance may already exist.

## 4. Run and confirm success

```bash
scripts/ec2.sh run
scripts/ec2.sh status
```

`run` starts a detached job and returns at once.
Repeat `status` every few minutes until its last line is `reproduce: ok` or `failed exit=...`; do not go on to step 5 while it says `running`.
A run on this instance takes about 40 minutes, and the instance stops itself 30 minutes after the run ends.
It installs Nix and the pinned environments, downloads and verifies the inputs and the null cache, runs the workflow, validates the figures and compares the outputs with the pinned baseline.
Leave `NULL_CACHE` unset.

A successful log contains both `baseline: ok` and `reproduce: ok`.
A failed run is not a successful reproduction; read the log with:

```bash
scripts/ec2.sh ssh 'tail -80 ~/run.log'
```

Fix the specific failure and run again only after the previous job has ended; completed work is reused.
If the log shows `STAGE pipeline ok` and `STAGE figures ok` and the baseline comparison then reports a difference, the workflow completed and the results disagree with the baseline: report exactly that.
Never relax a baseline check to turn a mismatch into success.

## 5. Pull results, then delete your resources

Once `status` ends with `reproduce: ok`:

```bash
scripts/ec2.sh pull "$RESULTS"
grep '^baseline: ok$' "$RESULTS/run.log"
grep '^reproduce: ok$' "$RESULTS/run.log"
```

Check that `pull` succeeded, and look at the saved log and the results under `data/processed/`, before deleting the instance.
Report the Git commit, the machine type, whether the workflow completed, whether baseline verification passed, and the elapsed time from the `started` line to `STAGE done` in `run.log`.
Then clean up with the same profile, region and name:

```bash
scripts/ec2.sh down --yes
```

This deletes the instance, its disk, its security group and its AWS key pair.
It also removes the local key files (`~/.ssh/$NAME`, `.pub` and `.known_hosts`, or the same under `KEY` if you set it) and the instance's address from `~/.ssh/known_hosts` if it is listed there.
Local files go to the trash when `trash` is installed and are deleted otherwise.
Keep the local results.

After a failed run, pull the logs and outputs needed for diagnosis, then clean up the same way; do not leave an abandoned run behind.
If cleanup is interrupted, run it again, then confirm that no instance, disk, security group or key pair of your run remains.

The instance stops itself 30 minutes after a run ends, and after 16 hours at the latest.
A stopped instance still bills for its disk and is not cleaned up; `down --yes` deletes it in that state too.
If it stopped before you pulled, run `scripts/ec2.sh start`, then pull and clean up.
If no GPU capacity is available to restart it, `scripts/ec2.sh rescue` starts it as a CPU machine for retrieving results only; do not run the workflow there.

## Optional: run on an existing GPU machine

This needs Linux, an NVIDIA L40S with a compatible driver, at least 32 CPUs, 248 GB RAM and 250 GB disk, plus Git and Nix ([official installer](https://nixos.org/download/)). Reopen the shell after installing Nix if asked to, and check `nix --version`.
Clone the repository as in step 3, then run from its root:

```bash
nix --extra-experimental-features 'nix-command flakes' develop --command just reproduce
```

To check results you already have without recomputing them:

```bash
nix --extra-experimental-features 'nix-command flakes' develop --command just verify-baseline /absolute/path/to/results
```

The path must contain `data/processed/`.

## Optional: Slurm without Nix or root

This route replaces Nix with a user-level pixi and conda-forge shell tools.
It has completed the workflow on an A100; read [outcomes on other GPU hardware](#outcomes-on-other-gpu-hardware) before interpreting its result.

You need:

- A single-node Linux x86-64 GPU allocation with at least 32 CPUs, the node's full memory and NVIDIA drivers compatible with the pinned CUDA environments.
  Request the CPUs with `--cpus-per-task`, which the commands below read.
  The workflow sizes its memory budget from host RAM, so request the node exclusively (`--exclusive` and `--mem=0` in [Slurm's allocation options](https://slurm.schedmd.com/sbatch.html)); the A100 run had 400 GB.
  Partition, account and GPU request syntax are site-specific.
- Outbound HTTPS from the compute node for packages and public inputs.
  If compute nodes cannot download, arrange a site-supported route first; an environment built without visible GPU drivers is not a validated substitute.
- Bash, Git, curl, coreutils and util-linux (`flock` and `setsid`).
- At least 250 GB on storage that outlives the batch job for the checkout, environments and outputs, plus home quota for the 25 GB null cache in `~/.copairs` and the package caches.
- `ulimit -Hn` of at least 65536.

Clone the repository and select the revision as in step 3, skipping the launcher commands.
Install the pixi binary pinned in `flake.nix`, checking its SHA256 before use, and the shell tools:

```bash
set -euo pipefail
mkdir -p scratch/bootstrap/bin
curl --proto '=https' --tlsv1.2 -fsSL \
    https://github.com/prefix-dev/pixi/releases/download/v0.68.0/pixi-x86_64-unknown-linux-musl \
    -o scratch/bootstrap/bin/pixi
printf '%s  %s\n' \
    "$(printf '%s' OpRgmobG7Bu2GW8TAw733m3omxQLfhFOX6zWwPoDL8g= | base64 -d | od -An -tx1 | tr -d ' \n')" \
    scratch/bootstrap/bin/pixi | sha256sum --check
chmod +x scratch/bootstrap/bin/pixi
export PATH="$PWD/scratch/bootstrap/bin:$HOME/.pixi/bin:$PATH"
pixi global install --environment jump-tools --channel conda-forge just rclone uv cmake ninja 'python=3.12'
```

These are [user-level shell tools](https://pixi.prefix.dev/latest/reference/cli/pixi/global/install/); analysis dependencies still come from the unchanged `pixi.lock`.
Run the rest inside the GPU job, from the checkout root, and not on a login node: environment builds probe CUDA, so setup belongs on the allocated GPU too.

```bash
set -euo pipefail
cd "$(pwd -P)"
export PATH="$PWD/scratch/bootstrap/bin:$HOME/.pixi/bin:$PATH"
export CORES="${SLURM_CPUS_PER_TASK:?Request CPUs per task in the Slurm job}"
export XDG_RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/jump-runtime.XXXXXX")"
chmod 700 "$XDG_RUNTIME_DIR"
git rev-parse HEAD
nvidia-smi
just reproduce 2>&1 | tee reproduce.log
```

Keep Slurm's `CUDA_VISIBLE_DEVICES`; do not replace it with GPU IDs from outside the allocation.
`cd "$(pwd -P)"` resolves symlinks so that the shell and Hydra agree on the checkout path.
`CORES` is explicit because `nproc --all` reports the whole node, which may exceed the allocation.

Read the outcome from `scratch/rerun/progress.log`, `reproduce.log` and the scheduler's exit status, as described in the next section.

## Outcomes on other GPU hardware

`just reproduce` reports its two outcomes separately:

| Outcome | Evidence | What it establishes |
| --- | --- | --- |
| Workflow completion | `STAGE pipeline ok` and `STAGE figures ok` in `scratch/rerun/progress.log` | Every job ran and every figure package validated |
| Baseline agreement | `baseline: ok`, followed by `reproduce: ok` and exit status 0 | The outputs match the L40S snapshot under the checks below |

The baseline was computed on an L40S, and some GPU computations give different results on other hardware.
A run on another GPU can therefore complete the workflow and then exit nonzero at the baseline comparison.
Report that as a completed workflow with a baseline mismatch, and keep the results.
Do not change tolerances, replace the baseline or disable the null cache to hide the difference.

What is known:

- **L40S**: both outcomes pass on a fresh `g6e.8xlarge`.
  The snapshot was computed on a `g6e.16xlarge`, and `g6e.12xlarge` is the launcher's other tested type.
- **A100 80 GB**: an independent collaborator reported completing every pipeline job and validating all six figure sets through the Slurm route with 32 CPUs and 400 GB RAM, in about 53 minutes including setup.
  The report describes differences in Harmony-corrected activity calls and GPU embeddings, and exact agreement for uncorrected CP and DL copairs outputs and for the batch-effect and MitoTox summaries.
  Those results have not been rechecked from that run's outputs.
- **A10G**: not supported for baseline agreement.
  Harmony-corrected mAP and activity calls varied materially, including between two A10G runs.
- Machines smaller than the 32-CPU, 248 GB `g6e.8xlarge` are untested.

After a change that touches only the verifier, `just verify-baseline` rechecks retained outputs without recomputing them.

## What is pinned and checked

| Pinned | By |
| --- | --- |
| Environment | `flake.lock` and `pixi.lock` |
| Inputs | `configs/inputs.sha256` |
| copairs null distributions | A manifest whose SHA256 is in the Justfile, specific to the copairs revision in `pyproject.toml` |
| Baseline snapshot [cached-l40s-20261115-04d51ad](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_snapshots/cached-l40s-20261115-04d51ad/README.md) | Its root manifest SHA256 in the Justfile |
| Deliberate differences from the snapshot | `configs/scientific_baseline.json`; see [the release contract](release.md#baseline-projection) |

The workflow restores the 25 GB null cache to `~/.copairs` and checks every file; existing files of the same name are replaced.
The snapshot is immutable.
The `04d51ad` in its name is a pre-publication revision that is not in this repository's history, which starts from a new root commit.

`just verify-baseline` downloads only the comparison files, authenticates them against the pinned manifest and fails on any of these:

- A missing or extra file under `data/processed/`.
  The required roster is the 2,002 retained snapshot paths plus the results catalog.
- A changed SHA256 among 998 exact artifacts.
- A difference in 146 Hydra configurations.
  Only the checkout root is normalized, in `runtime.cwd`, `runtime.output_dir` and file-schema `runtime.config_sources`; every other field must match.
- A missing, extra, duplicate or changed row among the 1,571 dataset-reference claims.
- A difference in the `numbers` of the batch-effect and MitoTox summaries.
- A difference in the columns, JCP2022 membership or values of the four curated compound tables.
- A results catalog, `data/processed/catalog.json`, that names a missing file or omits a summary envelope.
  The catalog postdates the snapshot and is compared with no baseline bytes.

| Compared values | Tolerance |
| --- | --- |
| Integers, strings and file hashes | Exact |
| Floating-point claim and summary values | Relative 1e-12, no absolute tolerance |
| `centroid_distance` in the curated tables | Absolute 2e-7, which bounds the measured float32 reduction variation |
| `fastest.name`, `fastest.transform_time_full_s`, and `transform_time_full_s` and `transform_time_sample_s` in each `transformations` entry, all from `fingerprint-metrics/fingerprint_metrics_summary.json` | Ignored: they are measured runtimes, and the fastest method can change between runs. The claims must still be present |

The snapshot's `provenance.json` names every file that is not compared by hash and why: rendering, timestamps, row order, database and container serialization, and duplicate summary wrappers.
Those files remain in the required roster.

### Differences from the published datastore

The pipeline uses exact GPU nearest neighbors, seeded random UMAP initialization and a seeded MitoTox permutation generator.
In repeated fixed-input experiments these gave identical graphs, embeddings and chemical viewport membership.
These are method changes from the run behind the published datastore, not tolerances around its numbers: exact neighbors change iLISI and kBET values, and the initialization changes layouts and UMAP stability claims.
CP still mixes better than DL and all kBET acceptance values remain low.
A reproduction therefore matches the snapshot's `CLAIMS.csv`, which is the authority for reproduced values, and not every number in the datastore, which stays unchanged.

### Running without the null cache

`NULL_CACHE=0` makes copairs compute its null distributions instead of restoring them, as a diagnostic:

```bash
NULL_CACHE=0 nix --extra-experimental-features 'nix-command flakes' develop --command just reproduce
```

Independently computed nulls can differ from the cached ones, so this run skips the baseline comparison, and its `reproduce: ok` confirms workflow completion and figure validation only.
It does not erase an existing null cache.
Set the variable on the GPU machine: the launcher does not forward it from the local terminal.

### Expected time

| Run | Time |
| --- | --- |
| Fresh `g6e.8xlarge` (one L40S), `started` to `STAGE done` | About 38 minutes |
| Fresh `g6e.12xlarge` (four L40S) | About 29 minutes |
| Without the null cache | About 3.6 hours in the latest run, 6 to 7 hours in earlier ones |

Fresh means no installed environment, null cache or outputs.
The times exclude account setup, quota approval, capacity waits and pulling results.
