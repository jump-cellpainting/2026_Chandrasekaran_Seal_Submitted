#!/usr/bin/env bash
# Collate profile comparison images into a single PDF
# Usage: ./scripts/1.01-collate-profile-comparison-report.sh
# Output: data/processed/profile-comparison/report.pdf

set -euo pipefail

BASE_DIR="data/processed/profile-comparison"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Directory not found: $BASE_DIR" >&2
    exit 1
fi

OUTPUT="${BASE_DIR}/report.pdf"
TMPDIR=$(mktemp -d)
trap 'if command -v trash >/dev/null 2>&1; then trash "$TMPDIR"; else printf "Temporary report images retained at %s\n" "$TMPDIR" >&2; fi' EXIT

# make_title TEXT OUTFILE WIDTH HEIGHT FONTSIZE
make_title() {
    python -c "
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
fig, ax = plt.subplots(figsize=(int(sys.argv[3])/100, int(sys.argv[4])/100), dpi=100)
ax.text(0.5, 0.5, sys.argv[1], ha='center', va='center', fontsize=int(sys.argv[5]), transform=ax.transAxes)
ax.axis('off'); fig.savefig(sys.argv[2], bbox_inches='tight', pad_inches=0.5, facecolor='white'); plt.close(fig)
" "$@"
}

IMAGES=()
slide_idx=0
for comparison_dir in "$BASE_DIR"/*/; do
    [[ -d "$comparison_dir" ]] || continue
    comparison_name=$(basename "$comparison_dir")

    title_png="${TMPDIR}/title_${slide_idx}.png"
    make_title "${comparison_name//_/ }" "$title_png" 1600 1200 28
    IMAGES+=("$title_png")
    slide_idx=$((slide_idx + 1))

    # Activity section (only if images exist)
    if [[ -f "${comparison_dir}activity/threshold_curve.png" ]]; then
        section_png="${TMPDIR}/title_${slide_idx}.png"
        make_title "Activity" "$section_png" 1600 60 32
        IMAGES+=("$section_png")
        slide_idx=$((slide_idx + 1))
        for img in threshold_curve.png overlap_upset.png; do
            [[ -f "${comparison_dir}activity/${img}" ]] && IMAGES+=("${comparison_dir}activity/${img}")
        done
    fi

    # Consistency sections
    # Structure: consistency/{group_type}/{distance}/
    if [[ -d "${comparison_dir}consistency" ]]; then
        for group_dir in "${comparison_dir}consistency"/*/; do
            [[ -d "$group_dir" ]] || continue
            group_name=$(basename "$group_dir")

            for distance_dir in "$group_dir"/*/; do
                [[ -d "$distance_dir" ]] || continue
                distance_name=$(basename "$distance_dir")

                section_png="${TMPDIR}/title_${slide_idx}.png"
                make_title "${group_name//_/ } (${distance_name})" "$section_png" 1600 60 32
                IMAGES+=("$section_png")
                slide_idx=$((slide_idx + 1))
                for img in threshold_curve.png overlap_upset.png; do
                    [[ -f "${distance_dir}${img}" ]] && IMAGES+=("${distance_dir}${img}")
                done
            done
        done
    fi
done

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "Error: No images found in $BASE_DIR" >&2
    exit 1
fi

echo "Collating images into $OUTPUT"
magick "${IMAGES[@]}" "$OUTPUT"
du -h "$OUTPUT"
