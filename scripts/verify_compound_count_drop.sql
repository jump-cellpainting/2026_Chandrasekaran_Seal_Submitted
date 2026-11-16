-- Verify the compound count drop from 115,796 (metadata) to 112,472 (copairs results).
-- Context: https://github.com/broadinstitute/jump_production/issues/40
--
-- The drop is explained by three factors:
--   1. 3,191 compounds exist only in source_7, which is excluded in the
--      compound_no_source7 dataset used for copairs analysis.
--   2. 124 compounds have profiles only on TARGET2 plates. The
--      activity_no_target2 preprocessing filters out TARGET2 plates
--      (Metadata_PlateType != 'TARGET2'), leaving these with zero wells.
--      All 124 were nominated by source_4 (per compound_source.Metadata_Compound_Source),
--      meaning source_4 selected these compounds for inclusion in JUMP.
--   3. 8 positive controls + 1 empty are excluded by the poscon filter
--      (Metadata_pert_type != 'poscon').
--
-- The final row (verification_check) returns the total if the three buckets
-- sum correctly, or -1 if something is off.
--
-- Usage: pixi run duckdb < scripts/verify_compound_count_drop.sql
ATTACH 'data/interim/jump_metadata_augmented.duckdb' AS meta (READ_ONLY);

ATTACH 'data/processed/copairs_results.duckdb' AS cop (READ_ONLY);

WITH
    total AS (
        SELECT
            COUNT(DISTINCT Metadata_JCP2022) AS n
        FROM
            meta.compound
    ),
    -- Compounds that only exist in source_7 (excluded by compound_no_source7 dataset)
    source7_only AS (
        SELECT
            COUNT(DISTINCT w.Metadata_JCP2022) AS n
        FROM
            meta.well w
            JOIN meta.plate p USING (Metadata_Plate, Metadata_Source)
            JOIN meta.compound c USING (Metadata_JCP2022)
        WHERE
            NOT EXISTS (
                SELECT
                    1
                FROM
                    meta.well w2
                    JOIN meta.plate p2 USING (Metadata_Plate, Metadata_Source)
                WHERE
                    w2.Metadata_JCP2022 = w.Metadata_JCP2022
                    AND p2.Metadata_Source != 'source_7'
            )
    ),
    -- Compounds in copairs activity_scores (primary downstream output)
    in_copairs AS (
        SELECT
            COUNT(DISTINCT Metadata_JCP2022) AS n
        FROM
            cop.activity_scores
        WHERE
            _dataset = 'compound_no_source7'
            AND _preprocessing = 'activity_no_target2'
    ),
    -- Remaining gap: compounds in non-source_7 wells but absent from copairs.
    -- This includes:
    --   - 124 compounds with profiles only on TARGET2 plates (filtered by activity_no_target2)
    --   - 8 positive controls (filtered by Metadata_pert_type != 'poscon')
    --   - 1 empty control
    gap AS (
        SELECT
            COUNT(DISTINCT c.Metadata_JCP2022) AS n
        FROM
            meta.compound c
        WHERE
            EXISTS (
                SELECT
                    1
                FROM
                    meta.well w
                    JOIN meta.plate p USING (Metadata_Plate, Metadata_Source)
                WHERE
                    w.Metadata_JCP2022 = c.Metadata_JCP2022
                    AND p.Metadata_Source != 'source_7'
            )
            AND NOT EXISTS (
                SELECT
                    1
                FROM
                    cop.activity_scores a
                WHERE
                    a.Metadata_JCP2022 = c.Metadata_JCP2022
                    AND a._dataset = 'compound_no_source7'
                    AND a._preprocessing = 'activity_no_target2'
            )
    )
SELECT
    'total_compounds' AS metric,
    total.n AS count
FROM
    total
UNION ALL
SELECT
    'source7_only_excluded',
    source7_only.n
FROM
    source7_only
UNION ALL
SELECT
    'target2_and_controls',
    gap.n
FROM
    gap
UNION ALL
SELECT
    'in_copairs_results',
    in_copairs.n
FROM
    in_copairs
UNION ALL
SELECT
    'verification_check',
    CASE
        WHEN total.n = source7_only.n + gap.n + in_copairs.n THEN total.n
        ELSE -1
    END
FROM
    total,
    source7_only,
    gap,
    in_copairs;