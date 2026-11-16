#!/usr/bin/env python
"""Download MitoTox database - compounds with mitochondrial toxicity labels.

Downloads from https://www.mitotox.org/api and fetches SMILES from PubChem.
Data is licensed under CC BY-NC 4.0.

Usage:
    python -m jump_production.downloading.mitotox
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR

app = typer.Typer()

# API endpoints
BASE_URL = "https://www.mitotox.org/api"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"

# Top-level mitochondrial function categories
TOP_LEVEL_FUNCTIONS = {
    "F01": "Transmembrane potential",
    "F02": "Function of mitochondria",
    "F03": "Organization of mitochondria",
    "F04": "Movement of mitochondria",
    "F05": "Oxidative stress",
    "F06": "Mitochondrial DNA",
    "F07": "Cell death",
    "F08": "Signaling",
}

OUTPUT_DIR = EXTERNAL_DATA_DIR / "mitotox"


def fetch_all_pages(url: str) -> list[dict]:
    """Fetch all pages from a paginated API endpoint."""
    results = []
    while url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        results.extend(data.get("results", []))
        url = data.get("next")
        time.sleep(0.3)
    return results


def load_or_fetch(cache_path: Path, url: str, name: str) -> list[dict]:
    """Load from cache or fetch from API."""
    if cache_path.exists():
        logger.info(f"Loading cached {name}")
        return json.loads(cache_path.read_text())

    logger.info(f"Fetching {name} from API...")
    data = fetch_all_pages(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data


def fetch_smiles(cache_path: Path, cids: list[str]) -> dict[str, str]:
    """Fetch SMILES from PubChem for a list of CIDs."""
    if cache_path.exists():
        logger.info("Loading cached SMILES")
        return json.loads(cache_path.read_text())

    cid_to_smiles = {}
    valid_cids = [c for c in cids if c and str(c).isdigit()]

    for i in range(0, len(valid_cids), 100):
        batch = valid_cids[i : i + 100]
        url = f"{PUBCHEM_URL}/{','.join(map(str, batch))}/property/CanonicalSMILES/JSON"
        try:
            logger.info(f"Fetching SMILES {i + 1}-{i + len(batch)} of {len(valid_cids)}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            for prop in response.json().get("PropertyTable", {}).get("Properties", []):
                smiles = prop.get("CanonicalSMILES") or prop.get("ConnectivitySMILES")
                cid_to_smiles[str(prop["CID"])] = smiles
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch SMILES batch {i}: {e}")
        time.sleep(0.3)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cid_to_smiles))
    return cid_to_smiles


@app.command()
def main(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    refresh: bool = typer.Option(False, "--refresh", "-r", help="Clear cache and re-download"),
) -> int:
    """Download MitoTox database and save to parquet."""
    output_path = OUTPUT_DIR / "mitotox_compounds.parquet"

    if output_path.exists() and not force:
        logger.info(f"Output exists: {output_path}. Use --force to overwrite.")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTPUT_DIR / "cache"

    if refresh and cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        logger.info("Cleared cache")

    compounds = load_or_fetch(cache_dir / "compounds.json", f"{BASE_URL}/compounds/list", "compounds")
    records = load_or_fetch(cache_dir / "records.json", f"{BASE_URL}/records/list", "records")

    # Build compound lookup
    compound_lookup = {
        c["compound_ID"]: {"name": c.get("name"), "cid": c.get("cid")} for c in compounds if c.get("compound_ID")
    }

    # Determine toxicity per compound (toxic if any record shows result=True)
    compound_toxicity: dict[str, dict] = {}
    for rec in records:
        compound_field = rec.get("compound", "")
        if ": " not in compound_field:
            continue
        compound_id = compound_field.split(": ")[0]

        if compound_id not in compound_toxicity:
            compound_toxicity[compound_id] = {"toxic": False, "functions": set()}

        if rec.get("result"):
            compound_toxicity[compound_id]["toxic"] = True
            func_code = rec.get("func") or ""
            if len(func_code) >= 3 and func_code[:3] in TOP_LEVEL_FUNCTIONS:
                compound_toxicity[compound_id]["functions"].add(TOP_LEVEL_FUNCTIONS[func_code[:3]])

    # Fetch SMILES from PubChem
    all_cids = [str(c.get("cid")) for c in compounds if c.get("cid")]
    cid_to_smiles = fetch_smiles(cache_dir / "smiles.json", all_cids)

    # Build dataframe
    rows = []
    for compound_id, info in compound_lookup.items():
        cid = info.get("cid")
        smiles = cid_to_smiles.get(str(cid)) if cid else None
        if not smiles:
            continue

        tox = compound_toxicity.get(compound_id, {"toxic": False, "functions": set()})
        rows.append(
            {
                "smiles": smiles,
                "compound_name": info.get("name"),
                "compound_id": compound_id,
                "cid": cid,
                "mitotox_label": "toxic" if tox["toxic"] else "non-toxic",
                "functional_mechanism": "|".join(sorted(tox["functions"])) or None,
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

    toxic_count = (df["mitotox_label"] == "toxic").sum()
    logger.success(f"Saved {len(df)} compounds ({toxic_count} toxic) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
