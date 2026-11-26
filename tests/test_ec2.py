"""Exercise launcher failure paths without credentials or cloud resources."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ec2.sh"
STUB = r"""
import json, os, pathlib, sys, time
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps([name, *args]) + '\n')
mode = os.environ.get('MODE', '')
state_file = pathlib.Path(os.environ['STATE'])
state = state_file.read_text()
if name == 'aws':
    command = args[1]
    if command == 'describe-instances':
        if mode == 'expired' and 'Name=instance-state-name,Values=running' in args:
            sys.stderr.write('ExpiredToken: refresh credentials\n')
            sys.exit(255)
        filters = args[args.index('--filters') + 2].split('Values=')[1].split(',')
        query = args[args.index('--query') + 1]
        if state not in filters:
            print('None')
        else:
            print(('None' if state == 'stopped' else '203.0.113.7') if query.endswith('PublicIpAddress') else 'i-test')
    elif command == 'terminate-instances':
        state_file.write_text('shutting-down')
    elif command == 'wait' and args[2] == 'instance-terminated':
        if mode == 'wait-fails':
            sys.stderr.write('ExpiredToken: waiter failed\n')
            sys.exit(255)
        state_file.write_text('terminated')
    elif command == 'describe-security-groups':
        print('None' if mode == 'missing' else 'sg-test')
    elif command == 'delete-security-group':
        if mode in ('missing', 'denied'):
            sys.stderr.write('InvalidGroup.NotFound' if mode == 'missing' else 'AccessDenied')
            sys.exit(255)
    elif command == 'describe-key-pairs':
        print('None' if mode == 'missing' else 'key-test')
    elif command == 'delete-key-pair':
        if mode == 'missing':
            sys.stderr.write('InvalidKeyPair.NotFound')
            sys.exit(255)
elif name == 'ssh':
    if mode == 'push-real':
        os.chdir(os.environ['HOME'])
        os.execv('/bin/bash', ['/bin/bash', '-c', args[-1]])
    if mode in ('timeout', 'expired', 'push-fails'):
        sys.stderr.write('ssh: connection refused\n')
        sys.exit(255)
    if 'cat >' in ' '.join(args):
        sys.stdin.buffer.read()
elif name == 'sleep':
    time.sleep(0.05)
elif name == 'git':
    if mode == 'push-real':
        if '-C' in args and pathlib.Path(args[args.index('-C') + 1]).is_absolute():
            args[args.index('-C') + 1] = os.environ['SOURCE_REPO']
        os.execv(os.environ['REAL_GIT'], ['git', *args])
    if 'bundle' in args:
        target = args[args.index('create') + 1]
        if target == '-':
            print('bundle')
        else:
            pathlib.Path(target).write_text('bundle')
    elif 'branch' in args:
        print('main')
elif name in ('trash', 'rm'):
    if mode == 'native-trash' and name == 'trash':
        os.execv('/usr/bin/trash', ['trash', *args])
    if mode == 'trash-fails' and name == 'trash':
        sys.stderr.write('trash failed\n')
        sys.exit(1)
    for arg in args:
        if arg not in ('--', '-f'):
            pathlib.Path(arg).unlink(missing_ok=True)
"""


@pytest.fixture(params=["environment", "system"])
def launcher(tmp_path, request):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for command in ("aws", "ssh", "scp", "git", "sleep", "ssh-keygen", "trash", "rm"):
        stub = bindir / command
        stub.write_text(f"#!{sys.executable}\n" + STUB)
        stub.chmod(0o755)
    shell = shutil.which("bash") if request.param == "environment" else "/bin/bash"
    if request.param == "system" and Path("/usr/bin/rsync").exists():
        (bindir / "rsync").symlink_to("/usr/bin/rsync")
    state = tmp_path / "state"
    state.write_text("running")
    calls = tmp_path / "calls"
    key = tmp_path / "custom user's key"
    key.write_text("fixture key")
    key.with_suffix(".pub").write_text("fixture public key")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh/known_hosts").write_text("fixture host")
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "HOME": str(home),
        "NAME": "test-repro",
        "KEY": str(key),
        "STATE": str(state),
        "CALLS": str(calls),
        "TMPDIR": str(temporary),
        "SSH_WAIT_SECONDS": "1",
        "REAL_GIT": shutil.which("git"),
        "SOURCE_REPO": str(tmp_path / "source.git"),
    }

    def run(command, mode="", instance_state="running", key_override=None):
        state.write_text(instance_state)
        run_env = {**env, "MODE": mode}
        if key_override is not None:
            run_env["KEY"] = key_override
        if mode == "no-trash":
            (bindir / "trash").rename(bindir / "disabled-trash")
            for tool in ("bash", "dirname"):
                (bindir / tool).symlink_to(shutil.which(tool))
            run_env["PATH"] = str(bindir)
        try:
            result = subprocess.run(
                [shell, str(SCRIPT), *command],
                cwd=tmp_path,
                env=run_env,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("launcher did not stop after SSH failure")
        events = [json.loads(line) for line in calls.read_text().splitlines()]
        return result, events

    return run, key, temporary


@pytest.mark.parametrize("state", ["running", "shutting-down"])
def test_down_waits_and_cleans_custom_key_and_host(launcher, state):
    run, key, _ = launcher
    result, calls = run(["down", "--yes"], instance_state=state)
    assert result.returncode == 0, result.stderr
    assert any(c[:4] == ["aws", "ec2", "wait", "instance-terminated"] for c in calls)
    assert ["ssh-keygen", "-R", "203.0.113.7"] in calls
    assert not key.exists()
    assert not key.with_suffix(".pub").exists()


def test_down_with_missing_resources_is_repeatable(launcher):
    run, key, _ = launcher
    for _ in range(2):
        result, _ = run(["down", "--yes"], mode="missing", instance_state="terminated")
        assert result.returncode == 0, result.stderr
    assert not key.exists()


def test_down_attempts_key_cleanup_after_group_error(launcher):
    run, key, _ = launcher
    result, calls = run(["down", "--yes"], mode="denied", instance_state="terminated")
    assert result.returncode != 0
    assert "AccessDenied" in result.stderr
    assert any(c[:3] == ["aws", "ec2", "delete-key-pair"] for c in calls)
    assert not key.exists()


def test_down_retains_keys_when_termination_wait_fails(launcher):
    run, key, _ = launcher
    result, calls = run(["down", "--yes"], mode="wait-fails")
    assert result.returncode != 0
    assert key.exists()
    assert not any(c[:3] == ["aws", "ec2", "delete-security-group"] for c in calls)


@pytest.mark.parametrize("mode", ["", "push-fails"])
def test_push_leaves_no_temporary_bundle(launcher, mode):
    run, _, temporary = launcher
    result, _ = run(["push", "main"], mode=mode)
    assert (result.returncode == 0) == (mode == "")
    assert list(temporary.iterdir()) == []


def test_push_stream_clones_and_updates_with_real_git(launcher):
    run, key, temporary = launcher
    source_repo = key.parent / "source.git"
    subprocess.run(
        ["git", "clone", "--bare", "--shared", "-q", str(SCRIPT.parents[1]), str(source_repo)],
        check=True,
    )
    branch = "ec2-test"
    subprocess.run(["git", "-C", str(source_repo), "branch", branch, "HEAD"], check=True)
    for _ in range(2):
        result, _ = run(["push", branch], mode="push-real")
        assert result.returncode == 0, result.stderr
    source = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", branch], text=True)
    target = subprocess.check_output(
        ["git", "-C", str(key.parent / "home/jump_production"), "rev-parse", "HEAD"],
        text=True,
    )
    assert source == target
    assert list(temporary.iterdir()) == []


@pytest.mark.parametrize("mode, message", [("timeout", "Timed out"), ("expired", "ExpiredToken")])
def test_ssh_wait_fails_promptly_and_explains_failure(launcher, mode, message):
    run, _, _ = launcher
    result, _ = run(["rescue"], mode=mode)
    assert result.returncode != 0
    assert message in result.stderr


def test_down_removes_per_run_host_keys_for_stopped_instance(launcher):
    run, key, _ = launcher
    known_hosts = Path(f"{key}.known_hosts")
    known_hosts.write_text("fixture host key from a previous public IP")
    result, _ = run(["down", "--yes"], instance_state="stopped")
    assert result.returncode == 0, result.stderr
    assert not known_hosts.exists()


def test_ssh_uses_per_run_host_keys_and_tolerates_brief_stalls(launcher):
    run, key, _ = launcher
    result, calls = run(["ssh", "true"])
    assert result.returncode == 0, result.stderr
    ssh = next(c for c in calls if c[0] == "ssh")
    assert f'UserKnownHostsFile="{key}.known_hosts"' in ssh
    assert "ServerAliveInterval=15" in ssh
    assert "ServerAliveCountMax=6" in ssh


def test_down_without_trash_removes_only_run_specific_files(launcher):
    run, key, _ = launcher
    files = [key, Path(f"{key}.pub"), Path(f"{key}.known_hosts"), Path(f"{key}.known_hosts.old")]
    for path in files:
        path.write_text("fixture")
    unrelated = key.parent / "unrelated-key"
    unrelated.write_text("keep")
    result, calls = run(["down", "--yes"], mode="no-trash", instance_state="stopped")
    assert result.returncode == 0, result.stderr
    assert all(not path.exists() for path in files)
    assert unrelated.read_text() == "keep"
    assert {c[-1] for c in calls if c[0] == "rm"} == {str(path) for path in files}


def test_failed_trash_does_not_fall_back_to_deletion(launcher):
    run, key, _ = launcher
    result, calls = run(["down", "--yes"], mode="trash-fails")
    assert result.returncode != 0
    assert key.exists()
    assert not any(c[0] == "rm" for c in calls)


@pytest.mark.skipif(not shutil.which("rsync"), reason="requires the launcher's rsync prerequisite")
def test_pull_preserves_key_and_known_hosts_arguments(launcher):
    run, key, temporary = launcher
    # Real rsync reaches the SSH stub, then fails because the stub sends no data.
    result, calls = run(["pull", str(temporary / "results")])
    assert result.returncode != 0
    ssh = next(c for c in calls if c[0] == "ssh")
    assert ssh[ssh.index("-i") + 1] == str(key)
    assert f'UserKnownHostsFile="{key}.known_hosts"' in ssh


@pytest.mark.parametrize(("mode", "expected"), [("", True), ("push-fails", False)])
def test_pull_fetches_paper_figures_only_when_the_machine_has_them(launcher, tmp_path, mode, expected):
    run, _, temporary = launcher
    stub = tmp_path / "bin/rsync"
    stub.unlink(missing_ok=True)
    stub.write_text(f"#!{sys.executable}\n" + STUB)
    stub.chmod(0o755)
    # The SSH stub answers the directory probe with success, or with failure in this mode.
    result, calls = run(["pull", str(temporary / "results")], mode=mode)
    assert result.returncode == 0, result.stderr
    assert any(c[0] == "ssh" and "paper_figures" in c[-1] for c in calls)
    sources = [arg for c in calls if c[0] == "rsync" for arg in c if "jump_production/./" in arg]
    assert any(source.endswith("data/processed") for source in sources)
    assert any(source.endswith("data/paper_figures") for source in sources) is expected


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/trash").exists(),
    reason="requires native macOS trash",
)
def test_down_with_native_macos_trash(launcher):
    run, key, _ = launcher
    result, _ = run(["down", "--yes"], mode="native-trash", instance_state="terminated")
    assert result.returncode == 0, result.stderr
    assert not key.exists()
    assert not Path(f"{key}.pub").exists()


def test_relative_key_cannot_be_interpreted_as_trash_option(launcher):
    run, key, _ = launcher
    relative = key.with_name("-key")
    key.rename(relative)
    Path(f"{key}.pub").rename(Path(f"{relative}.pub"))
    result, calls = run(["down", "--yes"], key_override=relative.name)
    assert result.returncode == 0, result.stderr
    assert not relative.exists()
    assert [c[1:] for c in calls if c[0] == "trash"] == [[str(relative)], [f"{relative}.pub"]]
