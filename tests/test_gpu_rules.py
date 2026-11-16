"""Enforce the two-part GPU contract on Snakemake rules.

A GPU rule is only correct if it does BOTH of these:

  1. declares ``resources: gpu=1``  -- Snakemake treats an *undeclared* resource as
     unlimited, so a rule that omits this escapes the ``--resources gpu=N`` cap entirely
     and can be scheduled dozens-at-once; and
  2. wraps its command in ``scripts/gpu_lease.sh`` -- Snakemake never assigns a device id,
     so an unwrapped job defaults to card 0 (that card OOMs, the rest idle).

Forgetting either half fails silently and only shows up as a GPU OOM mid-run. That is
exactly how ``profiles_to_anndata`` shipped using the rapids env with no gpu resource.
This test makes the pairing mechanical instead of something to remember.

A rule counts as a GPU rule if it takes a lease *or* runs in a GPU env. Keying off the
lease matters: it is the half that does not depend on guessing every env name that might
ever want a card (``srijit_2_36`` runs in a cheminformatics env, not a rapids one).
"""

import re
from pathlib import Path
from typing import NamedTuple

import pytest

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"

LEASE = "scripts/gpu_lease.sh"

# Envs whose whole point is the GPU. A rule invoking one of these is a GPU rule even if
# someone forgot the lease -- which is the case this half is here to catch.
GPU_ENVS = ("rapids", "rapids-cheminformatics", "deepchem")

# Rules that use a GPU env but do no GPU compute (pure numpy/sklearn that merely lives in
# that env). Listed explicitly so an exemption is a deliberate, reviewable act. A rule that
# takes a lease is never exempt -- if it leases a card, it must be capped.
CPU_ONLY_EXEMPTIONS = {
    "explore_0_02",  # synthetic law-of-large-numbers demo; sklearn + numpy only
}


class Rule(NamedTuple):
    smk: Path
    name: str
    body: str

    def uses_gpu_env(self) -> bool:
        return any(f"pixi run -e {env} " in self.body for env in GPU_ENVS)

    def takes_lease(self) -> bool:
        return LEASE in self.body

    def declares_gpu_resource(self) -> bool:
        # Deliberately not anchored to `resources:\n  gpu=1` -- that only matches when gpu
        # is the *first* key, so adding `mem_mb` above it would fail the test with a
        # "missing resources: gpu=1" message pointing at a gpu=1 that is plainly there.
        return bool(re.search(r"\bgpu\s*=\s*1\b", self.body))


def iter_rules():
    """Yield a Rule for every `rule`/`checkpoint` block in rules/*.smk."""
    for smk in sorted(RULES_DIR.glob("*.smk")):
        text = smk.read_text()
        # Split on top-level `rule <name>:` / `checkpoint <name>:` declarations.
        parts = re.split(r"^(?:rule|checkpoint)\s+(\w+):", text, flags=re.MULTILINE)
        # parts = [preamble, name1, body1, name2, body2, ...]
        for name, body in zip(parts[1::2], parts[2::2]):
            yield Rule(smk, name, body)


def gpu_rules():
    """Rules subject to the GPU contract: they lease a card, or they run in a GPU env."""
    for rule in iter_rules():
        if rule.takes_lease():
            yield rule  # leases a card -> must be capped, no exemption possible
        elif rule.uses_gpu_env() and rule.name not in CPU_ONLY_EXEMPTIONS:
            yield rule


GPU_RULES = list(gpu_rules())


def test_there_are_gpu_rules_to_check():
    """Guard against the finders silently matching nothing (which would pass vacuously)."""
    assert GPU_RULES, "no GPU rules found - has the shell invocation style changed?"
    assert any(r.takes_lease() for r in GPU_RULES), (
        f"no rule wraps {LEASE} - has the lease been renamed or dropped?"
    )


@pytest.mark.parametrize("rule", GPU_RULES, ids=lambda r: f"{r.smk.name}:{r.name}")
def test_gpu_rule_declares_resource_and_uses_lease(rule: Rule):
    missing = []
    if not rule.declares_gpu_resource():
        missing.append("resources: gpu=1")
    if not rule.takes_lease():
        missing.append(f"{LEASE} wrapper")

    why = "takes a GPU lease" if rule.takes_lease() else "uses a GPU env"
    assert not missing, (
        f"{rule.smk.name}: rule `{rule.name}` {why} but is missing {' and '.join(missing)}. "
        "A GPU rule needs both, or it escapes the --resources cap / pins itself to card 0. "
        f"If it does no GPU compute, drop the lease and add it to CPU_ONLY_EXEMPTIONS in "
        f"{Path(__file__).name}."
    )
