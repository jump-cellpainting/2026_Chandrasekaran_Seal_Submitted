"""Compute physicochemical properties and molecular embeddings for JUMP compounds.

This module provides subcommands for computing different molecular representations:
- properties: Physicochemical descriptors (MW, LogP, TPSA, etc.) for DB import
- morgan: Morgan fingerprints (2048-bit, radius 2) for structure similarity
- chemberta: ChemBERTa-77M-MLM embeddings (384-dim) for learned representations

Each subcommand can be run independently, allowing efficient pipeline orchestration.

Usage:
    # Compute physicochemical properties
    pixi run -e cheminformatics python -m jump_production.processing.compound_properties properties

    # Compute Morgan fingerprints
    pixi run -e cheminformatics python -m jump_production.processing.compound_properties morgan

    # Compute ChemBERTa embeddings (use dedicated chemberta env for CUDA support)
    pixi run -e chemberta python -m jump_production.processing.compound_properties chemberta --device cuda

    # Compute everything
    pixi run -e cheminformatics python -m jump_production.processing.compound_properties all

Outputs:
    - data/interim/compound_featurization/properties.parquet (imported by augment_metadata_db.sql)
    - data/interim/compound_featurization/morgan_fp.npz (Morgan fingerprints for UMAP/similarity)
    - data/interim/compound_featurization/chemberta_77m_mlm.npz (ChemBERTa embeddings)
"""

import sys

import duckdb
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress RDKit warnings (including deprecation)
RDLogger.DisableLog("rdApp.*")

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

# Paths - read from base DB, write to interim files
BASE_METADATA_DB = EXTERNAL_DATA_DIR / "jump_metadata.duckdb"
FEATURIZATION_DIR = INTERIM_DATA_DIR / "compound_featurization"
PROPERTIES_FILE = FEATURIZATION_DIR / "properties.parquet"
MORGAN_FP_FILE = FEATURIZATION_DIR / "morgan_fp.npz"
CHEMBERTA_FILE = FEATURIZATION_DIR / "chemberta_77m_mlm.npz"

# Model constants
CHEMBERTA_EMBED_DIM = 384  # ChemBERTa-77M-MLM output dimension (hidden_size=384)

app = typer.Typer(help="Compute molecular properties and embeddings for JUMP compounds.")


# =============================================================================
# Shared utilities
# =============================================================================


def load_smiles_from_db(sample: float = 1.0) -> pd.DataFrame:
    """Load SMILES from the base metadata database.

    Args:
        sample: Fraction of compounds to load (1.0 = all, 0.1 = 10%)

    Returns:
        DataFrame with Metadata_JCP2022 and Metadata_SMILES columns
    """
    if not BASE_METADATA_DB.exists():
        logger.error(f"Base metadata database not found: {BASE_METADATA_DB}")
        sys.exit(1)

    logger.info(f"Loading SMILES from {BASE_METADATA_DB}...")
    con = duckdb.connect(str(BASE_METADATA_DB), read_only=True)
    try:
        df = con.execute("""
            SELECT Metadata_JCP2022, Metadata_SMILES
            FROM compound
            WHERE Metadata_SMILES IS NOT NULL
        """).fetchdf()
    finally:
        con.close()

    logger.info(f"Loaded {len(df):,} compounds with SMILES")

    if sample < 1.0:
        n_sample = int(len(df) * sample)
        df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {len(df):,} compounds ({sample * 100:.0f}%)")

    return df


# =============================================================================
# Properties computation
# =============================================================================

# Create PAINS filter catalog once (expensive to create, cheap to use)
_PAINS_PARAMS = FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_PAINS_PARAMS)


def _compute_single_compound(smiles: str) -> dict:
    """Compute properties for a single compound (no fingerprint).

    Returns dict with properties. Fingerprints are computed separately in batch.
    """
    result = {
        "Metadata_MW": np.nan,
        "Metadata_LogP": np.nan,
        "Metadata_TPSA": np.nan,
        "Metadata_HBD": np.nan,
        "Metadata_HBA": np.nan,
        "Metadata_RotatableBonds": np.nan,
        "Metadata_NumRings": np.nan,
        "Metadata_NumHeavyAtoms": np.nan,
        "Metadata_NumAromaticRings": np.nan,
        "Metadata_FractionCSP3": np.nan,
        "Metadata_Lipinski_Violations": np.nan,
        "Metadata_QED": np.nan,
        "Metadata_MurckoScaffold": None,
        "Metadata_HasPAINS": None,
        "Metadata_ValidMol": False,
    }

    if pd.isna(smiles):
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result

    result["Metadata_ValidMol"] = True

    # Basic properties
    result["Metadata_MW"] = Descriptors.MolWt(mol)
    result["Metadata_LogP"] = Descriptors.MolLogP(mol)
    result["Metadata_TPSA"] = Descriptors.TPSA(mol)
    result["Metadata_HBD"] = Lipinski.NumHDonors(mol)
    result["Metadata_HBA"] = Lipinski.NumHAcceptors(mol)
    result["Metadata_RotatableBonds"] = Lipinski.NumRotatableBonds(mol)
    result["Metadata_NumRings"] = Descriptors.RingCount(mol)
    result["Metadata_NumHeavyAtoms"] = Lipinski.HeavyAtomCount(mol)
    result["Metadata_NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
    result["Metadata_FractionCSP3"] = Descriptors.FractionCSP3(mol)

    # Lipinski violations
    result["Metadata_Lipinski_Violations"] = sum(
        [
            result["Metadata_MW"] > 500,
            result["Metadata_LogP"] > 5,
            result["Metadata_HBD"] > 5,
            result["Metadata_HBA"] > 10,
        ]
    )

    # QED (drug-likeness score 0-1)
    try:
        result["Metadata_QED"] = QED.qed(mol)
    except Exception as e:
        logger.warning(f"QED computation failed for SMILES {smiles[:50]}...: {e}")

    # Murcko scaffold
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        result["Metadata_MurckoScaffold"] = Chem.MolToSmiles(scaffold)
    except Exception as e:
        logger.warning(f"Murcko scaffold computation failed for SMILES {smiles[:50]}...: {e}")

    # PAINS filter (using pre-created catalog)
    try:
        result["Metadata_HasPAINS"] = _PAINS_CATALOG.HasMatch(mol)
    except Exception as e:
        logger.warning(f"PAINS filter check failed for SMILES {smiles[:50]}...: {e}")

    return result


def compute_all_properties(
    smiles_list: list[str],
    num_workers: int = 0,
) -> pd.DataFrame:
    """Compute properties for all compounds, optionally in parallel.

    Args:
        smiles_list: List of SMILES strings
        num_workers: Number of worker processes. 0 = all CPUs, 1 = sequential (no multiprocessing)

    Returns:
        DataFrame with computed properties
    """
    from multiprocessing import Pool, cpu_count

    if num_workers <= 0:
        num_workers = cpu_count()

    if num_workers == 1:
        # Sequential mode - useful for debugging
        logger.info(f"Computing properties for {len(smiles_list):,} compounds (sequential)...")
        results = [_compute_single_compound(smiles) for smiles in smiles_list]
    else:
        # Parallel mode
        logger.info(f"Computing properties for {len(smiles_list):,} compounds ({num_workers} workers)...")
        # Use chunksize for better performance with large lists
        chunksize = max(1, len(smiles_list) // (num_workers * 4))
        with Pool(processes=num_workers) as pool:
            results = pool.map(_compute_single_compound, smiles_list, chunksize=chunksize)

    props_df = pd.DataFrame(results)

    n_valid = props_df["Metadata_ValidMol"].sum()
    n_pains = props_df["Metadata_HasPAINS"].sum()
    logger.info(f"Valid molecules: {n_valid:,}/{len(props_df):,} ({100 * n_valid / len(props_df):.1f}%)")
    if n_valid > 0 and n_pains > 0:
        logger.info(f"PAINS compounds: {n_pains:,} ({100 * n_pains / n_valid:.1f}% of valid)")

    return props_df


# =============================================================================
# Morgan fingerprints
# =============================================================================

# Module-level generator for fingerprints (created once per process)
_MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=2048)


def _compute_single_fingerprint(smiles: str) -> tuple[np.ndarray, bool]:
    """Compute Morgan fingerprint for a single compound.

    Returns tuple of (fingerprint array [2048], is_valid).
    """
    if pd.isna(smiles):
        return np.zeros(2048, dtype=np.uint8), False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=np.uint8), False

    fp = _MORGAN_GENERATOR.GetFingerprint(mol)
    return np.array(fp, dtype=np.uint8), True


def compute_fingerprints_batch(
    smiles_list: list[str],
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Morgan fingerprints in parallel using multiprocessing.

    Args:
        smiles_list: List of SMILES strings
        num_workers: Number of worker processes (0 = all CPUs, 1 = sequential)

    Returns:
        Tuple of (fingerprint array [N, 2048], valid mask [N])
    """
    from multiprocessing import Pool, cpu_count

    if num_workers <= 0:
        num_workers = cpu_count()

    if num_workers == 1:
        logger.info(f"Computing fingerprints for {len(smiles_list):,} compounds (sequential)...")
        results = [_compute_single_fingerprint(smiles) for smiles in smiles_list]
    else:
        logger.info(f"Computing fingerprints for {len(smiles_list):,} compounds ({num_workers} workers)...")
        chunksize = max(1, len(smiles_list) // (num_workers * 4))
        with Pool(processes=num_workers) as pool:
            results = pool.map(_compute_single_fingerprint, smiles_list, chunksize=chunksize)

    # Unpack results
    fp_array = np.array([r[0] for r in results], dtype=np.uint8)
    valid_mask = np.array([r[1] for r in results], dtype=bool)

    logger.info(f"Valid molecules for fingerprints: {valid_mask.sum():,}/{len(smiles_list):,}")
    logger.info(f"Fingerprint array shape: {fp_array.shape}")
    return fp_array, valid_mask


# =============================================================================
# ChemBERTa embeddings
# =============================================================================


def compute_chemberta_embeddings(
    smiles_list: list[str],
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ChemBERTa-77M-MLM embeddings using HuggingFace transformers.

    Uses CLS token embedding from the pretrained ChemBERTa model.
    Produces 384-dim embeddings.

    Args:
        smiles_list: List of SMILES strings
        device: Device to use ('cpu', 'cuda', 'cuda:0', etc.). Default 'cpu'.
        batch_size: Batch size for inference. Default 64.

    Returns:
        Tuple of (embeddings array [N, 384], valid mask [N])
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Check CUDA availability if requested
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    logger.info("Loading ChemBERTa-77M-MLM from HuggingFace...")
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device).eval()

    logger.info(f"Computing embeddings for {len(smiles_list):,} compounds on {device}...")

    embeddings = []
    valid_mask = np.ones(len(smiles_list), dtype=bool)

    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]

            try:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                # CLS token is first token
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
            except Exception as e:
                logger.warning(f"Batch {i // batch_size} failed: {e}")
                # Mark batch as invalid, append NaN (not zeros) to distinguish from valid zero vectors
                valid_mask[i : i + len(batch)] = False
                embeddings.append(np.full((len(batch), CHEMBERTA_EMBED_DIM), np.nan, dtype=np.float32))

            if (i // batch_size) % 100 == 0 and i > 0:
                logger.info(f"  Processed {i:,}/{len(smiles_list):,} compounds")

    embed_array = np.vstack(embeddings).astype(np.float32)
    logger.info(f"Valid embeddings: {valid_mask.sum():,}/{len(smiles_list):,}")

    return embed_array, valid_mask


# =============================================================================
# CLI Subcommands
# =============================================================================


@app.command()
def properties(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output file"),
    sample: float = typer.Option(1.0, "--sample", "-s", help="Fraction of compounds to process"),
    num_workers: int = typer.Option(0, "--workers", "-w", help="Number of workers (0 = all CPUs, 1 = sequential)"),
) -> None:
    """Compute physicochemical properties and save to parquet for DB import."""
    FEATURIZATION_DIR.mkdir(parents=True, exist_ok=True)
    if PROPERTIES_FILE.exists() and not force:
        logger.info(f"Output file already exists: {PROPERTIES_FILE}")
        logger.info("Use --force to overwrite.")
        sys.exit(0)

    df = load_smiles_from_db(sample)
    smiles_list = df["Metadata_SMILES"].tolist()

    props_df = compute_all_properties(smiles_list, num_workers=num_workers)
    props_df.insert(0, "Metadata_JCP2022", df["Metadata_JCP2022"].values)

    logger.info(f"Saving properties to {PROPERTIES_FILE}...")
    props_df.to_parquet(PROPERTIES_FILE, index=False)
    props_size_mb = PROPERTIES_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved properties: {len(props_df):,} rows ({props_size_mb:.1f} MB)")

    # Summary stats
    logger.info("=== Summary ===")
    for col in ["Metadata_ValidMol", "Metadata_HasPAINS"]:
        n_true = props_df[col].sum()
        logger.info(f"  {col}: {n_true:,} ({100 * n_true / len(props_df):.1f}%)")

    lipinski_dist = props_df["Metadata_Lipinski_Violations"].value_counts().sort_index()
    logger.info("  Lipinski violations distribution:")
    for violations, count in lipinski_dist.items():
        logger.info(f"    {int(violations)}: {count:,} ({100 * count / len(props_df):.1f}%)")

    logger.success("Done computing compound properties!")


@app.command()
def morgan(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output file"),
    sample: float = typer.Option(1.0, "--sample", "-s", help="Fraction of compounds to process"),
    num_workers: int = typer.Option(0, "--workers", "-w", help="Number of workers (0 = all CPUs, 1 = sequential)"),
) -> None:
    """Compute Morgan fingerprints (2048-bit, radius 2) and save to npz."""
    FEATURIZATION_DIR.mkdir(parents=True, exist_ok=True)
    if MORGAN_FP_FILE.exists() and not force:
        logger.info(f"Output file already exists: {MORGAN_FP_FILE}")
        logger.info("Use --force to overwrite.")
        sys.exit(0)

    df = load_smiles_from_db(sample)
    smiles_list = df["Metadata_SMILES"].tolist()

    fp_array, valid_mask = compute_fingerprints_batch(smiles_list, num_workers=num_workers)

    logger.info(f"Saving Morgan fingerprints to {MORGAN_FP_FILE}...")
    np.savez_compressed(
        MORGAN_FP_FILE,
        fingerprints=fp_array,
        valid_mask=valid_mask,
        jcp2022=df["Metadata_JCP2022"].values,
        # Metadata for reproducibility
        featurizer="morgan",
        radius=2,
        nbits=2048,
    )
    fp_size_mb = MORGAN_FP_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved Morgan fingerprints: {fp_array.shape} ({fp_size_mb:.1f} MB)")

    logger.success("Done computing Morgan fingerprints!")


@app.command()
def chemberta(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output file"),
    sample: float = typer.Option(1.0, "--sample", "-s", help="Fraction of compounds to process"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device ('cpu', 'cuda', 'cuda:0', etc.)"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size for inference"),
) -> None:
    """Compute ChemBERTa-77M-MLM embeddings (384-dim) and save to npz."""
    FEATURIZATION_DIR.mkdir(parents=True, exist_ok=True)
    if CHEMBERTA_FILE.exists() and not force:
        logger.info(f"Output file already exists: {CHEMBERTA_FILE}")
        logger.info("Use --force to overwrite.")
        sys.exit(0)

    df = load_smiles_from_db(sample)
    smiles_list = df["Metadata_SMILES"].tolist()

    embed_array, valid_mask = compute_chemberta_embeddings(smiles_list, device=device, batch_size=batch_size)

    logger.info(f"Saving ChemBERTa embeddings to {CHEMBERTA_FILE}...")
    np.savez_compressed(
        CHEMBERTA_FILE,
        embeddings=embed_array,
        valid_mask=valid_mask,
        jcp2022=df["Metadata_JCP2022"].values,
        # Metadata
        featurizer="chemberta-77m-mlm",
        embed_dim=CHEMBERTA_EMBED_DIM,
    )
    embed_size_mb = CHEMBERTA_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved ChemBERTa embeddings: {embed_array.shape} ({embed_size_mb:.1f} MB)")

    logger.success("Done computing ChemBERTa embeddings!")


@app.command("all")
def all_featurizers(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output files"),
    sample: float = typer.Option(1.0, "--sample", "-s", help="Fraction of compounds to process"),
    num_workers: int = typer.Option(0, "--workers", "-w", help="Number of workers (0 = all CPUs)"),
    include_chemberta: bool = typer.Option(False, "--compute-chemberta", help="Also compute ChemBERTa embeddings"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device for ChemBERTa"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size for ChemBERTa inference"),
) -> None:
    """Compute all featurizers (properties + Morgan FP, optionally ChemBERTa)."""
    FEATURIZATION_DIR.mkdir(parents=True, exist_ok=True)
    df = load_smiles_from_db(sample)
    smiles_list = df["Metadata_SMILES"].tolist()

    # Properties
    if PROPERTIES_FILE.exists() and not force:
        logger.info(f"Properties file exists, skipping: {PROPERTIES_FILE}")
    else:
        props_df = compute_all_properties(smiles_list, num_workers=num_workers)
        props_df.insert(0, "Metadata_JCP2022", df["Metadata_JCP2022"].values)
        props_df.to_parquet(PROPERTIES_FILE, index=False)
        logger.info(f"Saved properties: {PROPERTIES_FILE}")

    # Morgan fingerprints
    if MORGAN_FP_FILE.exists() and not force:
        logger.info(f"Morgan FP file exists, skipping: {MORGAN_FP_FILE}")
    else:
        fp_array, valid_mask = compute_fingerprints_batch(smiles_list, num_workers=num_workers)
        np.savez_compressed(
            MORGAN_FP_FILE,
            fingerprints=fp_array,
            valid_mask=valid_mask,
            jcp2022=df["Metadata_JCP2022"].values,
            featurizer="morgan",
            radius=2,
            nbits=2048,
        )
        logger.info(f"Saved Morgan fingerprints: {MORGAN_FP_FILE}")

    # ChemBERTa (optional)
    if include_chemberta:
        if CHEMBERTA_FILE.exists() and not force:
            logger.info(f"ChemBERTa file exists, skipping: {CHEMBERTA_FILE}")
        else:
            embed_array, embed_valid_mask = compute_chemberta_embeddings(
                smiles_list, device=device, batch_size=batch_size
            )
            np.savez_compressed(
                CHEMBERTA_FILE,
                embeddings=embed_array,
                valid_mask=embed_valid_mask,
                jcp2022=df["Metadata_JCP2022"].values,
                featurizer="chemberta-77m-mlm",
                embed_dim=CHEMBERTA_EMBED_DIM,
            )
            logger.info(f"Saved ChemBERTa embeddings: {CHEMBERTA_FILE}")

    logger.success("Done computing all featurizers!")


if __name__ == "__main__":
    sys.exit(app())
