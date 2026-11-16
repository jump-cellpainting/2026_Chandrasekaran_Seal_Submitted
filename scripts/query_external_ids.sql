-- Query: How many JUMP compounds have external database identifiers?
--
-- Source: mappings_pointers.csv from the MotiVE/jump_compound_annotator dataset
--   Zenodo: https://zenodo.org/records/18197517 (DOI: 10.5281/zenodo.18197517)
--   Code:   https://github.com/broadinstitute/monorepo/tree/main/libs/jump_compound_annotator
--
-- The pointers file maps JUMP InChIKeys to identifiers in ~39 external databases
-- via UniChem (https://www.ebi.ac.uk/unichem/) and MyChem (https://mychem.info/).
--
-- Database descriptions (from UniChem where available):
--   pubchem           - PubChem Compounds (CIDs). Normalized compound database from NCBI.
--   pubchem_dotf      - PubChem Depositor-Supplied (Tautomers/forms)
--   pubchem_tpharma   - PubChem Thomson Pharma
--   chembl            - ChEMBL. Bioactive drug-like small molecules from literature.
--   drugbank          - DrugBank. Drug + drug target (sequence, structure, pathway) data.
--   drugcentral       - DrugCentral. Drug information (active ingredients, MOA, indications).
--   chebi             - ChEBI. Manually curated ontology of chemical entities.
--   bindingdb         - BindingDB. Measured binding affinities for protein-drug interactions.
--   hmdb              - HMDB. Small molecule metabolites found in the human body.
--   kegg_ligand       - KEGG Ligand. Chemical compounds and reactions in biological pathways.
--   brenda            - BRENDA. Enzyme functional data from primary literature.
--   gtopdb            - Guide to Pharmacology (IUPHAR/BPS). Ligand-activity-target relationships.
--   pharmgkb          - PharmGKB. Pharmacogenomics knowledge (drug-gene-disease).
--   lincs             - LINCS. Library of Integrated Network-based Cellular Signatures.
--   probes_and_drugs  - Probes & Drugs. Chemical probes and drugs portal.
--   surechembl        - SureChEMBL. Chemistry extracted from patent full text.
--   fdasrs            - FDA SRS (UNII). Unique Ingredient Identifiers for regulated products.
--   rxnorm            - RxNorm. Normalized names for clinical drugs (NLM).
--   dailymed          - DailyMed. FDA-approved drug labeling.
--   clinicaltrials    - ClinicalTrials.gov. Clinical trial intervention identifiers.
--   comptox           - CompTox (EPA). Chemical safety and toxicity data.
--   actor             - ACToR (EPA). Aggregated Computational Toxicology Resource.
--   zinc              - ZINC. Commercially available compounds for virtual screening.
--   mcule             - Mcule. Purchasable compound database and screening platform.
--   emolecules        - eMolecules. Commercial compound availability search.
--   selleck           - Selleck Chemicals. Research chemical supplier catalog.
--   MedChemExpress    - MedChemExpress (MCE). Bioactive compound supplier catalog.
--   chemicalbook      - ChemicalBook. Chemical product catalog and properties.
--   nikkaji           - Nikkaji (JCSD). Japan Chemical Substance Dictionary.
--   lipidmaps         - LIPID MAPS. Lipid classification, structures, and quantitation.
--   swisslipids       - SwissLipids. Curated lipid biology knowledge resource.
--   pdb               - PDB. Protein Data Bank ligand identifiers.
--   CCDC              - CSD (Cambridge Structural Database). Crystal structures.
--   nmrshiftdb2       - NMRShiftDB. NMR spectra for organic structures.
--   metabolights      - MetaboLights (EMBL-EBI). Metabolomics experiments and metabolite data.
--   nih_ncc           - NIH NCC. NIH Clinical Collection.
--   atlas             - Expression Atlas (EMBL-EBI). Gene expression patterns.
--   recon             - Recon (Virtual Metabolic Human). Metabolic network reconstruction.
--   rhea              - Rhea. Expert-curated biochemical reactions knowledgebase.
--
-- Usage:
--   pixi run duckdb < scripts/query_external_ids.sql

ATTACH 'data/interim/jump_metadata_augmented.duckdb' AS meta (READ_ONLY);

.mode markdown
.header on

-- Per-database coverage via UNPIVOT
-- column01 is an empty unnamed column in the source CSV (artifact of pandas pivot export)
WITH pointers AS (
    SELECT * EXCLUDE (column01)
    FROM read_csv(
        'https://zenodo.org/records/18197517/files/mappings_pointers.csv',
        auto_detect=true,
        all_varchar=true
    )
),
jump_pointers AS (
    SELECT j.Metadata_JCP2022, p.*
    FROM meta.compound j
    LEFT JOIN pointers p ON j.Metadata_InChIKey = p.inchikey
),
total AS (
    SELECT COUNT(*) AS n FROM meta.compound
),
unpivoted AS (
    SELECT Metadata_JCP2022, database, external_id
    FROM jump_pointers
    UNPIVOT (
        external_id FOR database IN (
            pubchem, pubchem_dotf, pubchem_tpharma,
            chembl, drugbank, drugcentral, chebi, bindingdb,
            hmdb, kegg_ligand, brenda, gtopdb, pharmgkb, lincs,
            probes_and_drugs, surechembl, fdasrs, rxnorm, dailymed,
            clinicaltrials, comptox, actor,
            zinc, mcule, emolecules, selleck, "MedChemExpress", chemicalbook,
            nikkaji, lipidmaps, swisslipids, pdb, "CCDC", nmrshiftdb2,
            metabolights, nih_ncc, atlas, recon, rhea
        )
    )
)
SELECT
    database,
    COUNT(DISTINCT Metadata_JCP2022) AS n_compounds,
    ROUND(100.0 * COUNT(DISTINCT Metadata_JCP2022) / (SELECT n FROM total), 1) AS pct_of_jump
FROM unpivoted
WHERE external_id IS NOT NULL AND external_id != ''
GROUP BY database
ORDER BY n_compounds DESC;

-- Summary
WITH jump_pointers AS (
    SELECT j.Metadata_JCP2022, p.inchikey
    FROM meta.compound j
    LEFT JOIN read_csv(
        'https://zenodo.org/records/18197517/files/mappings_pointers.csv',
        auto_detect=true,
        all_varchar=true
    ) p ON j.Metadata_InChIKey = p.inchikey
)
SELECT
    COUNT(*) AS total_jump_compounds,
    COUNT(inchikey) AS with_any_external_id,
    ROUND(100.0 * COUNT(inchikey) / COUNT(*), 1) AS pct_with_id,
    COUNT(*) - COUNT(inchikey) AS without_external_id,
    ROUND(100.0 * (COUNT(*) - COUNT(inchikey)) / COUNT(*), 1) AS pct_without_id
FROM jump_pointers;

DETACH meta;
