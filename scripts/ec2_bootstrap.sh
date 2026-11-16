#!/usr/bin/env bash
# Runs on the rented machine, inside tmux, started by `scripts/ec2.sh run`.
# Installs Nix if missing, then runs the one reproduce command from this clone.
# The machine powers off (stops, disk kept) 30 minutes after this script ends, and
# MAX_HOURS after it starts whatever happens. Cancel the first with: sudo shutdown -c
set -uo pipefail
echo "started $(date -u +%FT%TZ)"  # `ec2.sh status` reads this, `reproduce: ok` and `failed exit=N`
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

sudo shutdown -c 2>/dev/null || true
sudo systemctl stop jump-deadman.timer 2>/dev/null || true
sudo systemd-run --quiet --unit=jump-deadman --on-active="${MAX_HOURS:-16}h" systemctl poweroff
# Timers do not survive a reboot (a machine that ran out of memory resets), so power off 30 minutes after any boot.
echo '@reboot root shutdown -h +30' | sudo tee /etc/cron.d/jump-poweroff >/dev/null
trap 'code=$?; [ "$code" = 0 ] || echo "failed exit=$code"; sudo shutdown -h +30' EXIT

if [[ ! -r "$HOME/.nix-profile/etc/profile.d/nix.sh" ]]; then
    curl --proto '=https' --tlsv1.2 -fsSL https://releases.nixos.org/nix/nix-2.35.4/install -o /tmp/nix-install
    sh /tmp/nix-install --no-daemon --yes --no-channel-add --no-modify-profile || exit 1
fi
. "$HOME/.nix-profile/etc/profile.d/nix.sh"
nix --extra-experimental-features 'nix-command flakes' develop --command just reproduce
