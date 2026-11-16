#!/usr/bin/env bash
# Rent one GPU machine on AWS for `just reproduce`, using aws, ssh and rsync.
# The machine is found by its Name tag; SSH host keys are stored beside the local key.
# Uses your AWS settings (AWS_PROFILE, AWS_REGION); the region needs a default VPC.
#
#   scripts/ec2.sh up              launch (tries each type in each zone), wait for ssh
#   scripts/ec2.sh push [BRANCH]   copy a branch of this checkout to the machine (default: current)
#   scripts/ec2.sh run             start `just reproduce` in tmux on the machine
#   scripts/ec2.sh status          progress, load, GPU; last line: running, reproduce: ok, or failed exit=N
#   scripts/ec2.sh ssh [CMD]       shell or one command
#   scripts/ec2.sh pull [DIR]      rsync results and logs here (default: scratch/ec2-results)
#   scripts/ec2.sh start           restart a stopped machine (it stops itself 30 minutes after each run and each boot)
#   scripts/ec2.sh rescue          stopped and no GPU capacity: restart as a small CPU box
#   scripts/ec2.sh down --yes      delete the machine, its disk, its firewall group and its key
set -euo pipefail

NAME="${NAME:-jump-repro}"
# The run is proven on g6e.12xlarge (48 CPUs, 372 GB, four 48 GB GPUs; 29 minutes) and on
# g6e.8xlarge (32 CPUs, 248 GB, one such GPU; 37 minutes). A 16-CPU, 64 GB machine runs out
# of memory at one job per CPU, so nothing smaller is tried.
TYPES="${TYPES:-g6e.12xlarge g6e.8xlarge g6e.16xlarge}"
DISK_GB="${DISK_GB:-250}"
MAX_HOURS="${MAX_HOURS:-16}"   # the machine powers itself off after this
KEY="${KEY:-$HOME/.ssh/$NAME}"
# Native macOS trash has no -- separator; absolute paths cannot become options.
[[ "$KEY" == /* ]] || KEY="$PWD/$KEY"
KNOWN_HOSTS="$KEY.known_hosts"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# IdentitiesOnly: offer this key alone, or an ssh agent holding several keys is refused first.
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
    -o "UserKnownHostsFile=\"$KNOWN_HOSTS\"" -o ConnectTimeout=20 -o BatchMode=yes
    -o ServerAliveInterval=15 -o ServerAliveCountMax=6)

field() {  # field STATES QUERY: one field of the tagged instance in one of those states
    aws ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=$1" \
        --query "Reservations[].Instances[] | [0].$2" --output text
}
host() {
    local ip
    ip=$(field running PublicIpAddress) || return
    [[ -n "$ip" && "$ip" != None ]] || { echo "$NAME has no running public IP." >&2; return 1; }
    echo "ubuntu@$ip"
}
wait_ssh() {
    local timeout="${SSH_WAIT_SECONDS:-300}" deadline address error
    [[ "$timeout" =~ ^[1-9][0-9]*$ ]] || { echo 'SSH_WAIT_SECONDS must be a positive integer.' >&2; return 1; }
    deadline=$((SECONDS + timeout))
    while :; do
        address=$(host) || return
        if error=$("${SSH[@]}" "$address" true 2>&1); then return 0; fi
        if (( SECONDS >= deadline )); then
            printf 'Timed out waiting for SSH at %s: %s\nCheck your key and security-group ingress (your public IP may have changed).\n' "$address" "$error" >&2
            echo 'The instance is still running. Fix access and continue with push, or clean up with down --yes; do not rerun up.' >&2
            return 1
        fi
        sleep 10
    done
}

case "${1:-}" in push|run|status|ssh|pull)
    [[ "$(field running InstanceId)" != None ]] ||
        { echo "$NAME is not running (it powers off after each run): scripts/ec2.sh start" >&2; exit 1; } ;;
esac
case "${1:-}" in
up)
    [[ "$(field pending,running,stopping,stopped InstanceId)" == None ]] || { echo "$NAME already exists." >&2; exit 1; }
    # One-time, idempotent: a dedicated key and a firewall group open to ssh from this address only.
    # If your address changes later, add the new one with authorize-security-group-ingress.
    [[ -f "$KEY" ]] || ssh-keygen -q -t ed25519 -N '' -f "$KEY"
    aws ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1 ||
        aws ec2 import-key-pair --key-name "$NAME" --public-key-material "fileb://$KEY.pub" >/dev/null
    sg=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" --query 'SecurityGroups[0].GroupId' --output text)
    if [[ "$sg" == None ]]; then
        sg=$(aws ec2 create-security-group --group-name "$NAME" --description "ssh for $NAME" --query GroupId --output text)
        aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 \
            --cidr "$(curl -fsS https://checkip.amazonaws.com)/32" >/dev/null
    fi
    # AWS-maintained Ubuntu 22.04 image with the NVIDIA driver; the pipeline sees its GPU as is.
    ami="${AMI:-$(aws ssm get-parameter --query Parameter.Value --output text \
        --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id)}"
    root=$(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].RootDeviceName' --output text)
    subnets=$(aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text)
    # Dead-man switch from first boot; ec2_bootstrap.sh re-arms it on every run.
    userdata=$(printf '#!/bin/sh\nsystemd-run --unit=jump-deadman --on-active=%sh systemctl poweroff\n' "$MAX_HOURS")
    for type in $TYPES; do for subnet in $subnets; do
        if out=$(aws ec2 run-instances --image-id "$ami" --instance-type "$type" --subnet-id "$subnet" \
                --key-name "$NAME" --security-group-ids "$sg" --user-data "$userdata" \
                --block-device-mappings "DeviceName=$root,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,Iops=16000,Throughput=1000,DeleteOnTermination=true}" \
                --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
                --query 'Instances[0].InstanceId' --output text 2>&1); then
            echo "Launched $out ($type, $subnet)"; break 2
        fi
        # Keep going only when the zone is full or does not offer this type; stop on anything else.
        grep -qE 'InsufficientInstanceCapacity|Unsupported' <<<"$out" || { echo "$out" >&2; exit 1; }
        echo "Not available: $type in $subnet"
    done; done
    [[ "${out:-}" == i-* ]] || { echo 'No capacity for any listed type in any zone.' >&2; exit 1; }
    aws ec2 wait instance-running --instance-ids "$out"; wait_ssh
    "${SSH[@]}" "$(host)" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | sed -n 2p; df -h / | tail -1'
    ;;
push)  # A git bundle gives the machine a true clone without a GitHub key on it. Later pushes fast-forward it.
    branch="${2:-$(git -C "$REPO" branch --show-current)}"
    address=$(host)
    git -C "$REPO" bundle create - "$branch" | "${SSH[@]}" "$address" "set -e
        cat > repo.bundle
        if [ -d jump_production ]; then git -C jump_production pull -q --ff-only ~/repo.bundle '$branch'
        else git clone -q -b '$branch' repo.bundle jump_production; fi
        git -C jump_production log --oneline -1"
    ;;
run)   "${SSH[@]}" "$(host)" "tmux new-session -d -s jump 'MAX_HOURS=$MAX_HOURS bash ~/jump_production/scripts/ec2_bootstrap.sh 2>&1 | tee -a ~/run.log'" ;;
status)
    "${SSH[@]}" "$(host)" 'cd ~/jump_production 2>/dev/null; tail -4 scratch/rerun/progress.log 2>/dev/null
        L=$(ls -t .snakemake/log/*.log 2>/dev/null | head -1)
        if [ -n "$L" ]; then
            echo "finished=$(grep -c "^Finished jobid" "$L") errors=$(grep -c "^Error in rule" "$L")"
            grep "^Error in rule" "$L" | sort | uniq -c
        fi
        uptime; free -g | sed -n 2p
        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
        if pgrep -f "^bash [^ ]*ec2_bootstrap.sh" >/dev/null; then echo running
        else case "$(grep -E "^(started |reproduce: ok|failed exit=)" ~/run.log 2>/dev/null | tail -1)" in
            "") echo "not started" ;; started*) echo "failed exit=unknown (the run was cut off)" ;;
            reproduce*) echo "reproduce: ok" ;; *) grep "^failed exit=" ~/run.log | tail -1 ;; esac; fi' ;;
ssh)   shift; "${SSH[@]}" "$(host)" "$@" ;;
pull)
    dest="${2:-$REPO/scratch/ec2-results}"; mkdir -p "$dest"
    # rsync parses its own command string; doubled quotes preserve literal quotes.
    ssh_command=""; quote="'"
    for arg in "${SSH[@]}"; do ssh_command+="'${arg//$quote/$quote$quote}' "; done
    rsync -a -e "$ssh_command" "$(host):run.log" "$dest/"
    rsync -a -e "$ssh_command" --relative "$(host):jump_production/./"{data/processed,logs,.snakemake/log,scratch/rerun} "$dest/" ;;
start)  # Restart a stopped machine. Its own type may have no capacity now, so try the others too.
    id=$(field stopped InstanceId)
    for type in $(field stopped InstanceType) $TYPES; do
        aws ec2 modify-instance-attribute --instance-id "$id" --instance-type "{\"Value\":\"$type\"}"
        if out=$(aws ec2 start-instances --instance-ids "$id" 2>&1); then started=$type; break; fi
        grep -qE 'InsufficientInstanceCapacity|Unsupported' <<<"$out" || { echo "$out" >&2; exit 1; }
        echo "Not available: $type"
    done
    [[ -n "${started:-}" ]] || { echo 'No GPU capacity in this zone; use rescue to read the disk.' >&2; exit 1; }
    aws ec2 wait instance-running --instance-ids "$id"; wait_ssh; echo "Up as $started." ;;
rescue)  # A stopped GPU machine may find no capacity on restart. Its disk does not need a GPU to be read.
    id=$(field stopped InstanceId)
    aws ec2 modify-instance-attribute --instance-id "$id" --instance-type '{"Value":"m6a.xlarge"}'
    aws ec2 start-instances --instance-ids "$id" >/dev/null
    aws ec2 wait instance-running --instance-ids "$id"; wait_ssh; echo "Up as m6a.xlarge: pull, then down." ;;
down)
    [[ "${2:-}" == --yes ]] || { echo 'This deletes the machine, its disk and its key. Pull first, then: down --yes' >&2; exit 1; }
    id=$(field pending,running,stopping,stopped,shutting-down InstanceId)
    cleanup_status=0
    # Skipped when the machine is already gone, so a second `down --yes` finishes a cleanup that failed.
    [[ "$id" == None ]] || {
        ip=$(field pending,running,stopping,stopped,shutting-down PublicIpAddress)
        aws ec2 terminate-instances --instance-ids "$id" --query 'TerminatingInstances[0].CurrentState.Name' --output text
        # Remove trust before the wait: a failed waiter must not lose the old address.
        if [[ -n "$ip" && "$ip" != None && -f "$HOME/.ssh/known_hosts" ]]; then
            ssh-keygen -R "$ip" >/dev/null || cleanup_status=1
        fi
        aws ec2 wait instance-terminated --instance-ids "$id"
    }
    # Also what `up` made for this name. The firewall group can go only once the machine is gone.
    if error=$(aws ec2 delete-security-group --group-name "$NAME" 2>&1); then :
    elif [[ "$error" != *'InvalidGroup.NotFound'* ]]; then
        echo "$error" >&2; cleanup_status=1
    fi
    if error=$(aws ec2 delete-key-pair --key-name "$NAME" 2>&1); then :
    elif [[ "$error" != *'InvalidKeyPair.NotFound'* ]]; then
        echo "$error" >&2; cleanup_status=1
    fi
    if command -v trash >/dev/null 2>&1; then cleanup=(trash)
    else cleanup=(rm -f --); fi
    for key_file in "$KEY" "$KEY.pub" "$KNOWN_HOSTS" "$KNOWN_HOSTS.old"; do
        [[ ! -e "$key_file" ]] || "${cleanup[@]}" "$key_file" || cleanup_status=1
    done
    exit "$cleanup_status" ;;
*)     sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
