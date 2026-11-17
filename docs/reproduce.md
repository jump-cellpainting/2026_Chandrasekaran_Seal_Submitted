# Reproduce on AWS

This guide is for a new user or agent with their own AWS account.
The recommended route uses a Mac or Linux computer to launch a Linux GPU instance; the pipeline runs on that instance.
Your local computer does not need a GPU, Nix, pixi, direnv, Kata, or any Broad credentials.
The bootstrap installs Nix on the instance, and the repository's pinned environment supplies the pipeline tools and Python environments.
Inputs, the null cache, and the comparison snapshot are public downloads; S3 write access is unnecessary.
If AWS is unavailable, use the [Slurm route](#optional-slurm-without-nix-or-root) below.
Workflow completion and agreement with the L40S baseline are separate outcomes; report both on other GPU hardware.

## 1. Prepare the computer that launches AWS

Use a terminal on macOS or Linux with Bash, Git, curl, OpenSSH (`ssh`, `ssh-keygen`), rsync, and AWS CLI v2.
On macOS, install the Command Line Tools with `xcode-select --install` if Git is missing; the other shell utilities are normally present.
On Ubuntu/Debian, install the shell tools with:

```bash
sudo apt-get update
sudo apt-get install -y git curl openssh-client rsync ca-certificates
```

Install AWS CLI v2 using [AWS's macOS or Linux installer](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). Check the required commands before launching:

```bash
for tool in bash git curl ssh ssh-keygen rsync aws; do
    command -v "$tool" || break
done
aws --version
mkdir -p ~/.ssh
```

Stop and install any missing command.
The computer needs outbound HTTPS for AWS and downloads, outbound SSH (port 22) to the instance, and several GB free for pulled outputs and logs.
The processed baseline is about 1.6 GB; raw inputs and the 25 GB null cache stay on the instance.

## 2. Prepare your AWS account

If you do not have an account, follow [AWS's account and CLI prerequisites](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-prereqs.html). This workflow creates billable GPU compute and disk resources in your account.
Configure your own named CLI profile using [AWS's authentication guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-authentication.html). For a console-based login, AWS CLI 2.32.0 or later supports the following example; use your own profile name if one is already configured:

```bash
export AWS_PROFILE=jump-reproduction
export AWS_REGION=us-west-2
aws login --profile "$AWS_PROFILE" --region "$AWS_REGION"
aws sts get-caller-identity
```

`aws login` opens a browser; use `--remote` on a computer without one.
Your IAM identity needs the login permission described in [AWS's login instructions](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html). If your organization uses IAM Identity Center, use its SSO profile and `aws sso login --profile "$AWS_PROFILE"` instead.
Confirm that `get-caller-identity` names the account you intend to use.
An expired session must be renewed before running launcher commands.
Temporary environment credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`) also work; give them enough lifetime for each command, because an already-running AWS waiter retains its original environment and termination can take several minutes (about six in a verified run).

Before launching, ensure that account has:

- EC2 permission to inspect images, instances, subnets, key pairs and security groups; launch and tag instances; import/delete key pairs; create/delete security groups and authorize their SSH ingress; and terminate instances (their attached disks are deleted automatically).
  Recovery commands also need permission to start instances and modify their instance type.
- At least 32 available vCPUs in the **Running On-Demand G and VT instances** quota in `us-west-2`, for the single-L40S `g6e.8xlarge` used below.
  New accounts may have zero; [request a quota increase](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html) before launching.
  Quota approval and available GPU capacity are separate requirements.
- A [default VPC and default public subnets](https://docs.aws.amazon.com/vpc/latest/userguide/default-vpc.html) in `us-west-2`, with public IP assignment and internet access.
  The launcher uses these existing subnets and creates an SSH security group restricted to your current public IP.
  It does not create a VPC or adapt to a private-only network.

If these prerequisites are unavailable, resolve them with the account administrator before proceeding.
Do not switch to an untested GPU or region to bypass them: the baseline uses L40S hardware and a region-specific image.

## 3. Clone and launch

All commands in this AWS path run directly in the local terminal, without a Nix shell or `direnv`.
Keep the same `AWS_PROFILE`, `AWS_REGION`, and `NAME` for the whole lifecycle, including cleanup; save their values if you change terminals.
Use a new name for each new run so it cannot select someone else's instance.

The public repository can be cloned without GitHub credentials.
The example selects `main`; to reproduce a particular published revision later, replace `main` with that revision's full commit ID.
Record the ID printed by `git rev-parse HEAD` with your run results.

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

`up` creates one L40S instance with 32 CPUs, 248 GB RAM and a 250 GB disk, plus a dedicated key pair and security group.
The tested Ubuntu image includes NVIDIA driver `595.112.07`; keep this AMI pin for baseline comparisons.
These checkout commands select the release revision and create a local branch for the launcher's Git bundle.
`push` transfers the current committed branch as a Git bundle, so the instance needs no GitHub credentials.
Uncommitted local edits are not transferred.
The script normally tries larger types too; `TYPES=g6e.8xlarge` keeps this example within the 32-vCPU quota.
If this type has no capacity, retry later or deliberately choose another tested L40S type after checking its quota: g6e.12xlarge needs 48 vCPUs, g6e.16xlarge needs 64.
If a launch fails after creating a key or security group, use the cleanup command in step 5 for that same name.
SSH readiness retries for five minutes (`SSH_WAIT_SECONDS=300`, plus the final connection attempt); on timeout, check the reported SSH error, your key, and whether your current public IP is still allowed by the security group.
The instance remains running after this timeout: fix access and continue with `scripts/ec2.sh push`, or use `scripts/ec2.sh down --yes` to clean up; rerunning `up` will report "already exists", and compute charges continue until the instance stops, with disk charges continuing until cleanup.

## 4. Run and confirm success

```bash
date -u +%FT%TZ
scripts/ec2.sh run
scripts/ec2.sh status
```

Record the time immediately before `run`.
`run` starts a detached job and returns; it does not wait for completion.
Repeat `status` until its last line is `reproduce: ok` or `failed exit=...`; do not run the next step while it still says `running`.
The first run installs Nix, enters the pinned environment, installs pipeline dependencies, downloads and verifies inputs and nulls, runs the workflow, validates figures, and compares outputs with the pinned baseline.
Nix is the remote environment manager; there is no separate local setup step for it.
The normal cached route is the default; leave `NULL_CACHE` unset.

A successful log contains both `baseline: ok` and `reproduce: ok`.
A failed run is not a successful reproduction; inspect the log with:

```bash
scripts/ec2.sh ssh 'tail -80 ~/run.log'
```

Fix the specific failure and resume only after the previous job has ended.
The workflow resumes completed work.
Do not relax baseline checks to turn a mismatch into success.

## 5. Pull results, then delete your resources

Once `status` reports success:

```bash
scripts/ec2.sh pull "$RESULTS"
grep '^baseline: ok$' "$RESULTS/run.log"
grep '^reproduce: ok$' "$RESULTS/run.log"
```

Confirm `pull` exited successfully and inspect the saved log and `data/processed/` results before deleting the instance.
Report the Git commit, machine type, elapsed time from the `started` timestamp to `STAGE done` in `run.log`, and whether baseline verification passed; `reproduce: ok` itself has no timestamp.
Then clean up using the same profile, region and name:

```bash
scripts/ec2.sh down --yes
```

This deletes that named instance, its disk, security group and AWS key pair, and removes the local key files selected by `KEY` (default: `~/.ssh/$NAME`) and their per-run host-key file (`$KEY.known_hosts`).
Local files go to the trash when `trash` is installed; otherwise they are deleted with `rm -f`.
SSH connections use that per-run host-key file, so cleanup also works after the instance has stopped and lost its public IP; for runs launched with an older script, cleanup removes the current IP from the shared `known_hosts` when AWS still reports it.
Keep the local results.
On failure, pull any logs or outputs needed for diagnosis before the same cleanup; do not leave an abandoned run behind.
If cleanup is interrupted, inspect what remains and finish removing only the resources belonging to your run.

The instance stops itself 30 minutes after a run ends, or after 16 hours at the latest; stopping retains the billed disk and is not cleanup.
If it has stopped before you pull, use `scripts/ec2.sh start`, then pull and delete it.
If GPU capacity prevents restarting, `scripts/ec2.sh rescue` starts the stopped instance as a CPU machine solely to retrieve results; do not run the GPU workflow there.

## Optional: run on an existing GPU machine

This alternative requires Linux, an NVIDIA L40S with a compatible driver, at least 32 CPUs, 248 GB RAM and 250 GB disk, plus Git and Nix.
Install Nix using [the official Linux instructions](https://nixos.org/download/), reopen the shell if required, and check `nix --version`.
Clone the repository as above, then run:

```bash
nix --extra-experimental-features 'nix-command flakes' develop --command just reproduce
```

The flake supplies pixi and the other pipeline tools; Kata and direnv are not required.
To check retained results on a machine with Nix, from the repository root use:

```bash
nix --extra-experimental-features 'nix-command flakes' develop --command just verify-baseline /absolute/path/to/results
```

The result directory must contain `data/processed/`.

## Optional: Slurm without Nix or root

An independent run on one A100 80 GB GPU, 32 CPUs and 400 GB RAM completed all 316 pipeline jobs and validated all six figure sets in about 53 minutes, including setup.
It used pixi 0.68.0 and conda-forge shell tools instead of Nix.
This establishes workflow compatibility, not exact agreement with the L40S baseline.
Harmony-corrected activity calls and GPU embeddings differed, while uncorrected CP/DL copairs outputs and the batch-effect and MitoTox scientific summaries matched.
Do not change scientific tolerances, replace the baseline or disable the null cache to conceal those differences.

Use a Linux x86-64 GPU allocation with sufficient disk space (at least 250 GB), NVIDIA drivers compatible with the pinned CUDA environments, and outbound HTTPS for package and input downloads.
The cluster must provide Bash, Git, curl, coreutils and util-linux (`flock` and `setsid`).
Keep the checkout, environments and outputs on storage that survives the batch job.
Use the same repository and revision selection from step 3; skip the AWS launcher commands.

Install the exact Linux pixi binary pinned in `flake.nix`, checking its SHA256 before execution:

```bash
mkdir -p scratch/bootstrap/bin
curl --proto '=https' --tlsv1.2 -fsSL \
    https://github.com/prefix-dev/pixi/releases/download/v0.68.0/pixi-x86_64-unknown-linux-musl \
    -o scratch/bootstrap/bin/pixi
printf '%s  %s\n' \
    "$(printf '%s' OpRgmobG7Bu2GW8TAw733m3omxQLfhFOX6zWwPoDL8g= | base64 -d | od -An -tx1 | tr -d ' \n')" \
    scratch/bootstrap/bin/pixi | sha256sum --check
chmod +x scratch/bootstrap/bin/pixi
export PATH="$PWD/scratch/bootstrap/bin:$HOME/.pixi/bin:$PATH"
pixi global install --environment jump-tools --channel conda-forge just rclone uv cmake ninja
```

These are [user-level shell tools](https://pixi.prefix.dev/latest/reference/cli/pixi/global/install/); analysis dependencies still come from the unchanged `pixi.lock`.
Run the following inside a single-node GPU job, after selecting your site's partition/account and requesting at least 32 CPUs and sufficient RAM (400 GB was used in the A100 run):

```bash
set -euo pipefail
export PATH="$PWD/scratch/bootstrap/bin:$HOME/.pixi/bin:$PATH"
export CORES="${SLURM_CPUS_PER_TASK:?Request CPUs per task in the Slurm job}"
export XDG_RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/jump-runtime.XXXXXX")"
chmod 700 "$XDG_RUNTIME_DIR"
git rev-parse HEAD
nvidia-smi
just reproduce 2>&1 | tee reproduce.log
```

Run from the checkout root and retain Slurm's `CUDA_VISIBLE_DEVICES`; do not replace it with GPU IDs from outside the allocation.
`CORES` is explicit because `nproc --all` reports the node's CPUs, which may exceed the allocation.
The workflow's memory budget uses host RAM, so use an exclusive node with its full memory allocation when the cluster does not expose allocation-limited memory in `/proc/meminfo`.
See [Slurm's allocation options](https://slurm.schedmd.com/sbatch.html) for `--cpus-per-task`, `--exclusive` and `--mem=0`; partition, account and GPU request syntax are site-specific.
Do not run the GPU workflow on a login node.
Environment builds can probe CUDA, so perform setup on the allocated GPU too.
If compute nodes cannot download packages or public data, arrange a site-supported network/setup route before starting; copying an environment built without visible drivers is not a validated substitute.

Inspect `scratch/rerun/progress.log`, `reproduce.log` and the scheduler exit status.
`STAGE pipeline ok` and `STAGE figures ok` establish workflow and figure validation; only a successful baseline comparison establishes agreement.
An A100 run may finish those stages and then exit nonzero on a genuine baseline difference.
Keep the results and report the mismatch rather than calling it `baseline: ok`.
After a verifier-only update, use `just verify-baseline` to check retained outputs without recomputing them.

## Timing and agreement contract

Before the source-release projection, fresh-machine cached runs took 37 minutes on one L40S and 29 minutes on four L40S GPUs (g6e.12xlarge).
The immutable baseline at scientific revision `04d51ad` passed 317 jobs and figure validation in 31 minutes 26 seconds on a single-L40S g6e.16xlarge, with Nix, RAPIDS and one verified input already warm and generated outputs absent.
That measurement includes 3 minutes 31 seconds of setup and is not a fresh-machine benchmark.
Account setup, quota approval, capacity waits and local result transfer are outside these timings.
Smaller machines are untested for the fast workflow.
A10G is not supported for baseline agreement: Harmony-corrected mAP and activity calls varied materially, including between two A10G runs.
All 877 copairs non-log artifacts agreed byte-for-byte across five earlier cached L40S runs and the current baseline.

The environment is pinned by `flake.lock` and `pixi.lock`, inputs by `configs/inputs.sha256`, and copairs null distributions by their pinned manifest.
The workflow restores the 25 GB null cache to `~/.copairs`, checking every file; existing files of the same name are replaced.
For an explicit uncached diagnostic on the GPU machine, `NULL_CACHE=0 nix --extra-experimental-features 'nix-command flakes' develop --command just reproduce` skips the restore and cached-baseline comparison.
It does not erase any existing null cache, and setting `NULL_CACHE` only in the local launcher terminal does not forward it to EC2.
Uncached `reproduce: ok` confirms workflow completion and figure validation only.
The latest fresh uncached run at `6684371` took 3 hours 35 minutes 54 seconds; earlier uncached runs took 6 to 7 hours.

The current immutable snapshot is [cached-l40s-20261115-04d51ad](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_snapshots/cached-l40s-20261115-04d51ad/README.md). Its root manifest SHA256 is pinned in the Justfile.
The verifier downloads only authenticated comparison inputs, applies the explicit source-release projection in `configs/scientific_baseline.json`, requires the complete 2,002-file processed roster, checks 998 exact artifact hashes and 146 Hydra configurations, and compares the full 1,571-row dataset-reference claim roster.
For Hydra, only `runtime.cwd` and the checkout-root prefixes of `runtime.output_dir` and file-schema `runtime.config_sources` are normalized.
Relative suffixes, configuration choices and every other field must match the authenticated baseline configuration exactly.
The original files and their absolute paths remain intact as run provenance.
The projection retires only three editorial audit outputs and ten non-scientific claim rows, replaces six provenance metadata values/keys, and pins two rewritten artifact hashes.
Every retained scientific value, output path and tolerance remains checked.
See [the release contract](release.md) for the migration and validation limits.
Missing, extra or duplicate claims fail.
The batch and MitoTox summary `numbers` objects are also checked, and all four curated compound tables require identical JCP2022 membership and non-distance fields.
Only `centroid_distance` permits absolute drift up to 2e-7, bounding the measured float32 reduction variation.
Claim and summary floating values allow relative roundoff up to 1e-12 with no absolute tolerance; integer counts remain exact.

The only ignored timing-derived claim values are `fastest.name`, `fastest.transform_time_full_s` and each `transformations` entry's `transform_time_full_s` and `transform_time_sample_s`, under source `data/processed/fingerprint-metrics/fingerprint_metrics_summary.json` and rule `fingerprint_metrics`.
The fastest method is selected by measured runtime, so its name can change across machines or repeated runs.
Timing keys must remain present and all other scientific values in those records are checked.
Snapshot `provenance.json` names every byte-check exclusion and reason, including rendering, timestamps, row order, database/container serialization and duplicate summary wrappers.
These exclusions do not remove files from the required roster.

The baseline intentionally adopts exact GPU neighbors and seeded random UMAP initialization, plus a local seeded MitoTox permutation generator.
Repeated fixed-input experiments demonstrated exact graphs, embeddings and chemical viewport membership at full relevant sizes, and the final production embeddings matched those experiments.
Exact neighbors change iLISI/kBET values and initialization changes layouts and UMAP stability claims; these are intentional method changes, not tolerances around historical numbers.
CP still mixes better than DL and all kBET acceptance values remain low.
The historical publication datastore remains separate and unchanged.
Dataset-reference reports are generated locally and are not source-release files.
The snapshot's `CLAIMS.csv` is the authority for current cached reproducibility; later verifier and documentation commits do not alter the scientific revision or its runtime evidence.

On 2026-11-16, pre-publication source revision `7404bb75a420e11dc673a7a6ed55c2d61a86bee8` completed a fresh-machine reproduction on a single-L40S g6e.8xlarge using the pinned AMI above.
The run took 37 minutes 26 seconds, from `started 2026-11-16T13:44:16Z` to `2026-11-16T14:21:42Z STAGE done`, including environment setup and verified input/cache downloads.
All 316 pipeline jobs, figure validation, 2,002 output paths, 1,144 exact artifact checks, 1,571 claims, four curated tables and both scientific summary comparisons passed.
The log contains `baseline: ok`, `reproduce: ok` and `STAGE exit=0`.
The run began with no installed Nix/pixi environment, null cache or generated outputs, and the remote source remained clean at the tested commit.
The public repository starts with a new signed root commit whose executable files, scientific configurations, environment pins and input pins are byte-identical to that tested revision.
Only documentation and its allowlist hashes were adapted for publication; the pre-publication revision is a provenance identifier, not a commit to check out from this fresh repository.

An earlier fresh run exposed unspecified well ordering before the cell-density source regression, which changed one Spearman correlation by about 4.6e-11 and failed an exact CSV hash.
The tested commit fixes the query order by source, plate, well and compound, restoring the existing baseline bytes without changing calculations, inputs, hashes or tolerances.
Local contract tests and projected-snapshot checks remain useful, but do not replace a fresh reproduction when executable source changes.
