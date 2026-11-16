-- Source 7 compound exclusivity and annotation analysis
-- Which source_7 compounds were profiled ONLY at source_7?
-- How does annotation coverage compare across imaging sources?
--
-- Usage: duckdb data/interim/jump_metadata_augmented.duckdb < scripts/query_source7_exclusivity.sql

-- Precompute compound-to-imaging-source mapping (used by queries 2-4)
CREATE TEMP TABLE compound_imaging_sources AS
SELECT
    w.Metadata_JCP2022,
    min(w.Metadata_Source) AS the_source,  -- meaningful only when n_imaging_sources = 1
    count(DISTINCT w.Metadata_Source) AS n_imaging_sources
FROM well w
JOIN perturbation p USING (Metadata_JCP2022)
WHERE p.Metadata_perturbation_modality = 'compound'
GROUP BY w.Metadata_JCP2022;

-- ============================================================
-- 1. Compounds per imaging source (all sources)
-- ============================================================
SELECT '--- Compounds per imaging source ---' AS section;

SELECT
    w.Metadata_Source,
    count(DISTINCT w.Metadata_JCP2022) AS n_compounds
FROM well w
JOIN perturbation p USING (Metadata_JCP2022)
WHERE p.Metadata_perturbation_modality = 'compound'
GROUP BY w.Metadata_Source
ORDER BY w.Metadata_Source;

-- ============================================================
-- 2. All compounds: distribution by number of imaging sources
-- ============================================================
SELECT '--- All compounds: how many sources each is profiled at ---' AS section;

SELECT
    n_imaging_sources,
    count(*) AS n_compounds,
    round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM compound_imaging_sources
GROUP BY n_imaging_sources
ORDER BY n_imaging_sources;

-- ============================================================
-- 3. Source 7 compounds: distribution by number of imaging sources
-- ============================================================
SELECT '--- Source 7 compounds: how many sources each is profiled at ---' AS section;

SELECT
    c.n_imaging_sources,
    count(*) AS n_compounds,
    round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
FROM compound_imaging_sources c
WHERE c.Metadata_JCP2022 IN (
    SELECT DISTINCT Metadata_JCP2022 FROM well WHERE Metadata_Source = 'source_7'
)
GROUP BY c.n_imaging_sources
ORDER BY c.n_imaging_sources;

-- ============================================================
-- 4. Single-source compounds: which source are they exclusive to?
-- ============================================================
SELECT '--- Single-source compounds: breakdown by source ---' AS section;

SELECT
    the_source,
    count(*) AS n_single_source_compounds
FROM compound_imaging_sources
WHERE n_imaging_sources = 1
GROUP BY the_source
ORDER BY n_single_source_compounds DESC;

-- ============================================================
-- 5. Annotation coverage by imaging source
-- ============================================================
SELECT '--- Annotation coverage (%) by imaging source ---' AS section;

WITH source_compounds AS (
    SELECT DISTINCT w.Metadata_Source, w.Metadata_JCP2022
    FROM well w
    JOIN perturbation p USING (Metadata_JCP2022)
    WHERE p.Metadata_perturbation_modality = 'compound'
),
annotated_compounds AS (
    SELECT DISTINCT Metadata_JCP2022, 'repurposing' AS annotation FROM repurposing_hub_annotations
    UNION ALL
    SELECT DISTINCT Metadata_JCP2022, 'chembl' AS annotation FROM chembl_protein_targets
    UNION ALL
    SELECT DISTINCT Metadata_JCP2022, 'chemical_probes' AS annotation FROM chemical_probes
    UNION ALL
    SELECT DISTINCT Metadata_JCP2022, 'motive' AS annotation FROM motive_annotations
    UNION ALL
    SELECT DISTINCT Metadata_JCP2022, 'toxcast' AS annotation FROM toxcast_annotations
)
SELECT
    sc.Metadata_Source AS source,
    count(DISTINCT sc.Metadata_JCP2022) AS n_compounds,
    round(100.0 * count(DISTINCT CASE WHEN a.annotation = 'repurposing' THEN sc.Metadata_JCP2022 END) / count(DISTINCT sc.Metadata_JCP2022), 1) AS pct_repurposing,
    round(100.0 * count(DISTINCT CASE WHEN a.annotation = 'chembl' THEN sc.Metadata_JCP2022 END) / count(DISTINCT sc.Metadata_JCP2022), 1) AS pct_chembl,
    round(100.0 * count(DISTINCT CASE WHEN a.annotation = 'chemical_probes' THEN sc.Metadata_JCP2022 END) / count(DISTINCT sc.Metadata_JCP2022), 1) AS pct_chem_probes,
    round(100.0 * count(DISTINCT CASE WHEN a.annotation = 'motive' THEN sc.Metadata_JCP2022 END) / count(DISTINCT sc.Metadata_JCP2022), 1) AS pct_motive,
    round(100.0 * count(DISTINCT CASE WHEN a.annotation = 'toxcast' THEN sc.Metadata_JCP2022 END) / count(DISTINCT sc.Metadata_JCP2022), 1) AS pct_toxcast
FROM source_compounds sc
LEFT JOIN annotated_compounds a ON a.Metadata_JCP2022 = sc.Metadata_JCP2022
GROUP BY sc.Metadata_Source
ORDER BY sc.Metadata_Source;

-- ============================================================
-- 6. Any-annotation coverage by imaging source (ranked)
-- ============================================================
SELECT '--- Compounds with ANY annotation, by imaging source ---' AS section;

WITH source_compounds AS (
    SELECT DISTINCT w.Metadata_Source, w.Metadata_JCP2022
    FROM well w
    JOIN perturbation p USING (Metadata_JCP2022)
    WHERE p.Metadata_perturbation_modality = 'compound'
),
any_annotated AS (
    SELECT DISTINCT Metadata_JCP2022 FROM repurposing_hub_annotations
    UNION SELECT DISTINCT Metadata_JCP2022 FROM chembl_protein_targets
    UNION SELECT DISTINCT Metadata_JCP2022 FROM chemical_probes
    UNION SELECT DISTINCT Metadata_JCP2022 FROM motive_annotations
    UNION SELECT DISTINCT Metadata_JCP2022 FROM toxcast_annotations
)
SELECT
    sc.Metadata_Source AS source,
    count(DISTINCT sc.Metadata_JCP2022) AS n_compounds,
    count(DISTINCT CASE WHEN aa.Metadata_JCP2022 IS NOT NULL THEN sc.Metadata_JCP2022 END) AS n_annotated,
    round(100.0 * count(DISTINCT CASE WHEN aa.Metadata_JCP2022 IS NOT NULL THEN sc.Metadata_JCP2022 END) / count(DISTINCT sc.Metadata_JCP2022), 1) AS pct_any_annotation
FROM source_compounds sc
LEFT JOIN any_annotated aa ON aa.Metadata_JCP2022 = sc.Metadata_JCP2022
GROUP BY sc.Metadata_Source
ORDER BY pct_any_annotation DESC;
