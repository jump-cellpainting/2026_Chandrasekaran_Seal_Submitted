"""Drive index.html in a real browser against output trees written by the real producers.

The page reads a public bucket directly, so these tests answer its bucket requests from a local
tree instead: envelopes come from `write_summary`, the catalog from `build_catalog`. That covers
the whole path a reader depends on, from what an analysis emits to what the page shows.

Playwright is not part of the pinned environments. Run with `just test-browser`; under plain
`just test` this module is skipped. Set JUMP_BROWSER_LIVE=1 to also check the page against the
published bucket.
"""

import contextlib
import json
import mimetypes
import os
import re
import shutil
from datetime import UTC, datetime
from email.utils import formatdate
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from xml.sax.saxutils import escape

import pytest

playwright_api = pytest.importorskip("playwright.sync_api", reason="needs Playwright; run `just test-browser`")

from jump_production.catalog import build_catalog  # noqa: E402
from jump_production.summaries import index_files, write_summary  # noqa: E402

INDEX = Path(__file__).resolve().parents[1] / "index.html"
ORIGIN = "https://explorer.test"
BUCKET = "https://cellpainting-gallery.s3.amazonaws.com"
PUBLICATION = "cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/"
DATASTORE = f"{PUBLICATION}jump_production_datastore/"
PROCESSED = f"{DATASTORE}processed/"
REVISION = "0123456789abcdef0123456789abcdef01234567"
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cfc0f01f0005000"
    "1ff89993d1d0000000049454e44ae426082"
)
MARKUP = '<img src=x onerror="window.injected=1"> probes'
REGISTRY = {
    "schema_version": 1,
    "repository": "https://example.org/source",
    "sections": [
        {"id": "figures", "title": "Paper figures", "description": "Rendered figures with their legends."},
        {"id": "activity", "title": "Phenotypic activity", "description": "Which compounds are active."},
        {"id": "quality", "title": "Data quality", "description": "Checks on the measurements."},
    ],
    "entries": [
        {
            "summary": "data/processed/figures/figure_collection/summary.json",
            "section": "figures",
            "title": "Compound collection figure",
        },
        {
            "summary": "data/processed/figures/figure_optional_heatmap/summary.json",
            "section": "figures",
            "title": "Positive-control heatmap",
            "optional": True,
            "highlights": [{"key": "matched_pairs", "label": "Compound-source pairs"}],
        },
        {
            "summary": "data/processed/activity-summary/summary.json",
            "section": "activity",
            "title": "Phenotypic activity headline",
            "highlights": [
                {"key": "analyzed_compounds", "label": "Compounds analyzed"},
                {"key": "rates.active_percent", "label": "Active rate"},
            ],
        },
        {
            "summary": "data/processed/kinase-probes/compound_no_source7/summary.json",
            "section": "activity",
            "title": "Kinase probe libraries",
        },
        {
            "path": "data/processed/dataset_reference/",
            "requires": "data/processed/dataset_reference/dataset_reference.json",
            "section": "activity",
            "title": "Reported numbers in one table",
            "description": "Every reported number in one table.",
            "files": ["dataset_reference.json"],
        },
        {
            "path": "data/processed/results.duckdb",
            "section": "activity",
            "title": "Results database",
            "description": "Every run in one database.",
        },
        {
            "summary": "data/processed/chemical-space/all/chemical_space_summary.json",
            "section": "quality",
            "title": "Chemical space of the collection",
        },
        {
            "summary": "data/processed/quality/compound_no_source7/summary.json",
            "section": "quality",
            "title": "Plate-edge quality checks",
        },
    ],
}
DECLARATIONS = [
    {"rule": "figure", "output": "data/processed/figures/{name}/summary.json", "script": "notebooks/figure.py"},
    {"rule": "activity", "output": "data/processed/activity-summary/summary.json", "script": "notebooks/0.13-activity.py"},
    {"rule": "kinase", "output": "data/processed/kinase-probes/{dataset}/summary.json", "script": "notebooks/2.44-kinase.py"},
    {"rule": "reference", "output": "data/processed/dataset_reference/dataset_reference.json", "script": None},
    {"rule": "store", "output": "data/processed/results.duckdb", "script": None},
    {"rule": "space", "output": "data/processed/chemical-space/all/chemical_space_summary.json", "script": None},
    {"rule": "quality", "output": "data/processed/quality/{dataset}/summary.json", "script": "notebooks/2.02-quality.py"},
]


def build_outputs(root: Path) -> Path:
    """Write a small checkout-shaped output tree through the real envelope writer."""
    processed = root / "data/processed"

    figure = processed / "figures/figure_collection"
    figure.mkdir(parents=True)
    (figure / "panel.png").write_bytes(PNG)
    (figure / "panel.pdf").write_bytes(b"%PDF-1.4\n")
    (figure / "source_table.csv").write_text("compound,value\nA,1\n")
    (figure / "legend.md").write_text(
        "# figure_collection: Compound collection\n\n## Collection overview (`panel.png`)\n\n"
        "Circle sizes are **schematic**. See [the paper](https://example.org/paper).\n"
    )
    write_summary(
        figure / "summary.json",
        description="Schematic of the compound collection.",
        numbers={"panels": 1},
        files=[
            {"path": "panel.png", "caption": "Collection overview"},
            *[item for item in index_files(figure, relative_to=figure) if item["path"] != "panel.png"],
            {"path": "legend.md", "kind": "data", "caption": "Figure legend"},
        ],
    )

    activity = processed / "activity-summary"
    activity.mkdir()
    write_summary(
        activity / "summary.json",
        description="Compounds tested for activity and the subset that is active.",
        numbers={
            "dataset": "compound_no_source7",
            "activity_threshold": 0.05,
            "analyzed_compounds": 112462,
            "phenotypically_active_compounds": 17078,
            "rates": {"active_percent": 15.2},
            "profile_object": {"size_bytes": 2766251888},
        },
        files=[],
    )

    kinase = processed / "kinase-probes/compound_no_source7"
    kinase.mkdir(parents=True)
    (kinase / "kinase_heatmap.png").write_bytes(PNG)
    (kinase / "probe_overlap.csv").write_text("library,count\nKCGS,49\n")
    write_summary(
        kinase / "summary.json",
        description=f"Overlap of kinase libraries with the collection. {MARKUP}",
        numbers={"task": "Kinase probe analysis", "binding_cutoff": 80, "n_in_collection": {"KCGS": 49}},
        files=[
            "data/processed/kinase-probes/compound_no_source7/kinase_heatmap.png",
            "data/processed/kinase-probes/compound_no_source7/probe_overlap.csv",
        ],
    )

    space = processed / "chemical-space/all"
    space.mkdir(parents=True)
    (space / "scaffolds.png").write_bytes(PNG)
    write_summary(
        space / "chemical_space_summary.json",
        description="Chemical-space characterization of the collection.",
        numbers={"total_compounds": 115796},
        files=["data/processed/chemical-space/all/scaffolds.png"],
    )

    quality = processed / "quality/compound_no_source7"
    (quality / "edge_effects").mkdir(parents=True)
    (quality / "edge_effects/edge_heatmap.png").write_bytes(PNG)
    # The workflow runs from the checkout root with relative paths, so the index is `data/...`.
    with contextlib.chdir(root):
        nested = index_files(Path("data/processed/quality/compound_no_source7/edge_effects"))
    write_summary(
        quality / "summary.json",
        description="Whether plate-edge wells differ from interior wells.",
        numbers={"n_wells": 558928},
        files=nested,
    )

    reference = processed / "dataset_reference"
    reference.mkdir()
    rows = [
        {"claim": "phenotypically_active_compounds", "value": 17078,
         "source": "data/processed/activity-summary/summary.json", "rule": "activity", "how": "numbers"},
        {"claim": "n_in_collection.KCGS", "value": 49,
         "source": "data/processed/kinase-probes/compound_no_source7/summary.json", "rule": "kinase", "how": "numbers"},
        {"claim": "total_compounds", "value": 115796,
         "source": "data/processed/chemical-space/all/chemical_space_summary.json", "rule": "space", "how": "numbers"},
    ]
    (reference / "dataset_reference.json").write_text(json.dumps(rows))
    (processed / "results.duckdb").write_bytes(b"duckdb\n")
    return root


def build_opt_in_figure(processed: Path) -> None:
    """Write the opt-in figure package the way its renderer does: bare names, a legend, a table."""
    figure = processed / "figures/figure_optional_heatmap"
    figure.mkdir(parents=True)
    (figure / "heatmap.png").write_bytes(PNG)
    (figure / "heatmap.csv").write_text("source,compound,value\nsource_1,C1,0.5\n")
    (figure / "legend.md").write_text("# figure_optional_heatmap: Heatmap\n\n## Heatmap (`heatmap.png`)\n\nValues per source.\n")
    write_summary(
        figure / "summary.json",
        description="Positive-control reproducibility across sources",
        numbers={"dataset": "compound_no_source7", "compounds": 8, "matched_pairs": 71},
        files=["heatmap.png", "heatmap.csv", "legend.md"],
    )


def publish(root: Path, bucket_root: Path, *, catalog: bool) -> None:
    """Lay the output tree out as the public datastore, with or without a catalog."""
    target = bucket_root / "jump_production_datastore/processed"
    shutil.copytree(root / "data/processed", target)
    if catalog:
        document = build_catalog(
            REGISTRY, root=root, declarations=DECLARATIONS,
            canonical_key={"default_dataset": "compound_no_source7"}, revision=REVISION,
        )
        (target / "catalog.json").write_text(json.dumps(document))
    profiles = bucket_root / "jump_production_datastore/raw/profiles"
    profiles.mkdir(parents=True)
    (profiles / "compound_DL_CPCNN_no_source7_rsc.parquet").write_bytes(b"PAR1")
    with (profiles / "compound_no_source7.parquet").open("wb") as stream:
        stream.truncate(300 * 1024 * 1024)  # sparse: listed at 300 MB, never read
    (bucket_root / "jump_production_datastore/README.md").write_text("# Datastore\n\nInputs and results.\n")


class Bucket:
    """Answer the page's bucket requests (ListObjectsV2 and GET) from a local directory."""

    def __init__(self, directory: Path, page_size: int = 4):
        self.page_size = page_size  # small pages, so continuation is always exercised
        self.objects = {
            PUBLICATION + path.relative_to(directory).as_posix(): path
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        self.requests: list[str] = []

    def listing(self, query: dict) -> str:
        prefix = query.get("prefix", [""])[0]
        delimiter = query.get("delimiter", [""])[0]
        limit = min(int(query.get("max-keys", [self.page_size])[0]), self.page_size)
        start = int(query.get("continuation-token", ["0"])[0])
        items: list[tuple[str, str]] = []
        for key in sorted(key for key in self.objects if key.startswith(prefix)):
            rest = key[len(prefix):]
            if delimiter and delimiter in rest:
                common = prefix + rest.split(delimiter)[0] + delimiter
                if ("prefix", common) not in items:
                    items.append(("prefix", common))
            else:
                items.append(("object", key))
        more = start + limit < len(items)
        body = [f"<ListBucketResult><IsTruncated>{'true' if more else 'false'}</IsTruncated>"]
        if more:
            body.append(f"<NextContinuationToken>{start + limit}</NextContinuationToken>")
        for kind, key in items[start:start + limit]:
            if kind == "prefix":
                body.append(f"<CommonPrefixes><Prefix>{escape(key)}</Prefix></CommonPrefixes>")
                continue
            stat = self.objects[key].stat()
            modified = datetime.fromtimestamp(stat.st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            body.append(
                f"<Contents><Key>{escape(key)}</Key><LastModified>{modified}</LastModified>"
                f"<Size>{stat.st_size}</Size></Contents>"
            )
        return "".join([*body, "</ListBucketResult>"])

    def handle(self, route) -> None:
        url = urlsplit(route.request.url)
        self.requests.append(route.request.url)
        cors = {"Access-Control-Allow-Origin": "*"}
        if url.path in ("", "/"):
            route.fulfill(status=200, headers={**cors, "Content-Type": "application/xml"}, body=self.listing(parse_qs(url.query)))
            return
        path = self.objects.get(unquote(url.path.lstrip("/")))
        if path is None:
            route.fulfill(status=404, headers={**cors, "Content-Type": "application/xml"}, body="<Error/>")
            return
        route.fulfill(
            status=200,
            body=path.read_bytes(),
            headers={
                **cors,
                "Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "Last-Modified": formatdate(path.stat().st_mtime, usegmt=True),
            },
        )


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as playwright:
        launched = None
        for options in ({"channel": "chrome"}, {}):
            try:
                launched = playwright.chromium.launch(headless=True, **options)
                break
            except playwright_api.Error:
                continue
        if launched is None:
            pytest.skip("needs Chrome or a Playwright Chromium (`playwright install chromium`)")
        yield launched
        launched.close()


@pytest.fixture(scope="module")
def outputs(tmp_path_factory) -> Path:
    return build_outputs(tmp_path_factory.mktemp("outputs"))


class Session:
    """One browser page whose bucket is a local tree; fails the test on any page error."""

    def __init__(self, browser, bucket: Bucket | None, index_text: str | None = None):
        self.bucket = bucket
        index_text = index_text or INDEX.read_text()
        self.page = browser.new_page(viewport={"width": 1280, "height": 900})
        self.problems: list[str] = []
        self.page.on("pageerror", lambda error: self.problems.append(f"pageerror: {error}"))
        self.page.on(
            "console",
            lambda message: self.problems.append(f"console: {message.text}") if message.type == "error" else None,
        )
        self.page.route(
            f"{ORIGIN}/**",
            lambda route: route.fulfill(status=200, content_type="text/html", body=index_text),
        )
        if bucket is not None:
            self.page.route(f"{BUCKET}/**", bucket.handle)
            self.page.route(f"{BUCKET}/?**", bucket.handle)

    def open(self, query: str = ""):
        self.page.goto(f"{ORIGIN}/index.html{query}", wait_until="networkidle")
        return self.page

    def settle(self):
        self.page.wait_for_load_state("networkidle")
        self.page.wait_for_timeout(350)  # debounced address-bar update
        return self.page

    def header(self) -> tuple[str, str, str]:
        return tuple(self.page.inner_text(selector) for selector in ("#location-label", "#folder-title", "#directory-summary"))


@pytest.fixture
def cataloged(browser, outputs, tmp_path):
    publish(outputs, tmp_path, catalog=True)
    session = Session(browser, Bucket(tmp_path))
    yield session
    session.page.close()
    assert not session.problems


@pytest.fixture
def uncataloged(browser, outputs, tmp_path):
    """The published state before any catalog exists, with the defects found in the live bucket."""
    publish(outputs, tmp_path, catalog=False)
    processed = tmp_path / "jump_production_datastore/processed"
    kinase = processed / "kinase-probes/compound_no_source7/summary.json"
    document = json.loads(kinase.read_text())
    for record in document["files"]:  # as written by an older run on someone else's machine
        record["path"] = f"/work/users/someone/checkout/{record['path']}"
    kinase.write_text(json.dumps(document))
    space = processed / "chemical-space/all/chemical_space_summary.json"
    document = json.loads(space.read_text())
    document["files"] = [{"path": "data/processed/chemical-space/scaffolds.png", "kind": "figure", "caption": "scaffolds"}]
    space.write_text(json.dumps(document))
    session = Session(browser, Bucket(tmp_path))
    yield session
    session.page.close()
    assert not session.problems


def texts(page, selector: str) -> list[str]:
    return [" ".join(text.split()) for text in page.locator(selector).all_inner_texts()]


def test_overview_lists_every_cataloged_result_under_its_section(cataloged):
    page = cataloged.open()

    assert cataloged.header() == (
        "PUBLISHED RESULTS", "Overview", "7 result sets in 3 sections | results from source revision 0123456",
    )
    assert texts(page, ".section-heading h3") == ["Start here", "Paper figures", "Phenotypic activity", "Data quality"]
    assert texts(page, "#section-activity .entry-title") == [
        "Phenotypic activity headline", "Kinase probe libraries", "Reported numbers in one table", "Results database",
    ]
    assert "Rendered figures with their legends." in page.inner_text("#section-figures")
    # One listing to find the catalog and one request to read it: no folder walking.
    assert len(cataloged.bucket.requests) == 2
    reproduce = page.locator(".folder-card", has_text="Reproduce the results").get_attribute("href")
    assert reproduce == f"https://example.org/source/blob/{REVISION}/docs/reproduce.md"


def test_search_finds_analyses_files_and_reported_numbers(cataloged):
    page = cataloged.open()
    page.fill("#search", "kinase")
    cataloged.settle()

    assert texts(page, ".entry-title") == ["Kinase probe libraries"]
    assert texts(page, ".manifest-caption") == ["kinase_heatmap.png"]
    assert texts(page, ".claim-row") == ["49 n_in_collection.KCGS Kinase probe libraries"]
    assert page.url == f"{ORIGIN}/index.html?q=kinase"

    page.fill("#search", "17078")
    cataloged.settle()
    assert texts(page, ".entry-title") == []
    assert texts(page, ".claim-row") == ["17,078 phenotypically_active_compounds Phenotypic activity headline"]

    # Leaving straight after typing still records the search, so Back returns to it.
    page.fill("#search", "kinase")
    page.click(".entry-card")
    cataloged.settle()
    assert cataloged.header()[:2] == ("ANALYSIS", "Kinase probe libraries")
    page.go_back()
    cataloged.settle()
    assert page.url == f"{ORIGIN}/index.html?q=kinase"
    assert page.input_value("#search") == "kinase" and texts(page, ".entry-title") == ["Kinase probe libraries"]


def test_analysis_page_shows_authored_title_headlines_and_source(cataloged):
    page = cataloged.open(f"?prefix={PROCESSED}activity-summary/")

    assert cataloged.header()[:2] == ("ANALYSIS", "Phenotypic activity headline")
    # Headlines are the catalog's curated keys, including a nested one, with authored labels.
    # A key ending in "percent" is shown with a percent sign, so the label does not repeat the unit.
    assert texts(page, ".summary-metric") == ["112,462 Compounds analyzed", "15.2% Active rate"]
    assert "Workflow rule activity" in page.inner_text(".summary-meta-row")
    source = page.locator(".summary-card-actions a", has_text="Analysis source").get_attribute("href")
    assert source == f"https://example.org/source/blob/{REVISION}/notebooks/0.13-activity.py"

    # Figures an analysis keeps in a subfolder are previewed on the analysis page itself.
    page = cataloged.open(f"?prefix={PROCESSED}quality/compound_no_source7/")
    assert cataloged.header()[:2] == ("ANALYSIS", "Plate-edge quality checks")
    assert texts(page, ".image-name") == ["edge_effects/edge_heatmap.png"]
    assert page.locator(".image-card img").evaluate("image => image.complete && image.naturalWidth") == 1

    # A folder that only contains analyses says so; a profile-set folder name is spelled out.
    page = cataloged.open(f"?prefix={PROCESSED}")
    cards = dict(zip(texts(page, ".folder-name"), texts(page, ".folder-action")))
    assert cards["activity-summary"] == "Phenotypic activity headline"
    assert cards["kinase-probes"] == "1 result set inside"
    page = cataloged.open(f"?prefix={PROCESSED}kinase-probes/")
    assert texts(page, ".folder-action") == ["Kinase probe libraries"]


def test_summary_document_opens_declared_outputs_in_both_path_conventions(cataloged):
    # A publication package declares paths relative to itself.
    page = cataloged.open(f"?file={PROCESSED}figures/figure_collection/summary.json")
    assert cataloged.header()[:2] == ("DOCUMENT", "summary.json")
    assert page.inner_text(".json-document-title") == "Compound collection figure"
    assert texts(page, ".manifest-caption") == ["Collection overview", "source table", "Figure legend"]
    assert texts(page, ".manifest-item:nth-child(1) .manifest-actions > *") == ["Preview", "PDF"]
    page.click(".manifest-item:nth-child(1) button")
    assert page.locator("#dialog-image").evaluate("image => image.decode().then(() => image.naturalWidth)") == 1
    assert page.locator("#dialog-image").get_attribute("src") == f"{BUCKET}/{PROCESSED}figures/figure_collection/panel.png"
    page.click("#dialog-close")
    table = page.locator(".manifest-item:nth-child(2) a").get_attribute("href")
    assert table.startswith("https://lite.datasette.io/?csv=") and table.endswith("#/data/source_table")
    assert unquote(table).count(f"{BUCKET}/{PROCESSED}figures/figure_collection/source_table.csv") == 1

    # Most analyses declare paths relative to the checkout.
    page = cataloged.open(f"?file={PROCESSED}kinase-probes/compound_no_source7/summary.json")
    assert texts(page, ".manifest-path") == ["kinase_heatmap.png", "probe_overlap.csv"]
    assert page.locator(".missing-note").count() == 0
    page.click(".manifest-item:nth-child(1) button")
    assert page.locator("#dialog-image").evaluate("image => image.decode().then(() => image.naturalWidth)") == 1


def test_large_tables_are_offered_as_downloads_not_browser_previews(cataloged):
    page = cataloged.open(f"?prefix={DATASTORE}raw/profiles/")
    rows = {row.locator(".file-name").inner_text(): row for row in page.locator(".file-row").all()}

    large = rows["compound_no_source7.parquet"]
    assert texts(large, ".file-actions > *") == ["Download", "Copy S3 path"]
    assert "CellProfiler features, Source 7 excluded" in large.inner_text()
    assert "Too large to open in the browser" in large.inner_text()
    assert large.locator(".file-name").get_attribute("href") == f"{BUCKET}/{DATASTORE}raw/profiles/compound_no_source7.parquet"

    small = rows["compound_DL_CPCNN_no_source7_rsc.parquet"]
    assert texts(small, ".file-actions > *") == ["Open file", "Explore data"]
    assert "Deep-learning (CPCNN) features, Source 7 excluded, Harmony re-run by batch" in small.inner_text()

    # A cataloged file that is not a folder opens its folder filtered to that file.
    page = cataloged.open()
    page.click(".entry-card:has-text('Results database')")
    cataloged.settle()
    assert texts(page, ".file-name") == ["results.duckdb"]
    assert page.url == f"{ORIGIN}/index.html?prefix={PROCESSED}&filter=results.duckdb"
    assert "DuckDB database" in page.inner_text(".file-row")


def test_legends_notes_and_text_files_read_in_the_page(cataloged):
    page = cataloged.open(f"?prefix={PROCESSED}figures/figure_collection/")
    assert cataloged.header()[:2] == ("PAPER FIGURE", "Compound collection figure")
    legend = page.locator(".section", has_text="Figure legends")
    assert texts(legend, ".prose h3, .prose h4") == ["figure_collection: Compound collection", "Collection overview (panel.png)"]
    assert legend.locator(".prose strong").inner_text() == "schematic"
    assert legend.locator(".prose a").get_attribute("href") == "https://example.org/paper"
    # The authored caption replaces the file-stem caption on the preview.
    assert texts(page, ".image-caption") == ["Collection overview"]

    page.click(".file-row:has-text('legend.md') .button-summary")
    cataloged.settle()
    assert cataloged.header()[:2] == ("DOCUMENT", "legend.md")
    assert "Circle sizes are schematic." in page.inner_text(".prose")

    page = cataloged.open(f"?prefix={DATASTORE}")
    assert "Inputs and results." in page.locator(".section", has_text="About this folder").inner_text()


def test_reported_numbers_are_searchable_and_link_to_their_summary(cataloged):
    page = cataloged.open("?view=numbers")
    assert cataloged.header() == ("PUBLISHED RESULTS", "Reported numbers", "3 values from 3 analyses, as written by each analysis")

    page.fill("#search", "chemical space")
    cataloged.settle()
    assert texts(page, ".claim-row") == ["115,796 total_compounds Chemical space of the collection"]
    assert page.url == f"{ORIGIN}/index.html?view=numbers&q=chemical+space"
    page.click(".claim-analysis")
    cataloged.settle()
    assert page.inner_text(".json-document-title") == "Chemical space of the collection"


def test_published_text_is_never_interpreted_as_markup(cataloged):
    page = cataloged.open(f"?prefix={PROCESSED}kinase-probes/compound_no_source7/")
    assert MARKUP in page.inner_text(".summary-description")
    page = cataloged.open("?q=kinase")
    assert MARKUP in page.inner_text(".entry-description")
    page = cataloged.open(f"?file={PROCESSED}kinase-probes/compound_no_source7/summary.json")
    assert MARKUP in page.inner_text(".json-document-description")
    assert page.locator("img[src='x']").count() == 0
    assert page.evaluate("window.injected") is None


def test_addresses_outside_the_publication_fall_back_to_the_overview(cataloged):
    for query in ("?prefix=cpg0016-jump/source_1/", f"?prefix={PUBLICATION}../", "?file=other-bucket/secret.json",
                  f"?file={PROCESSED}results.duckdb"):
        cataloged.open(query)
        assert cataloged.header()[:2] == ("PUBLISHED RESULTS", "Overview"), query
    assert all(PUBLICATION in unquote(request) for request in cataloged.bucket.requests)

    page = cataloged.open(f"?file={PROCESSED}missing/summary.json")
    assert "could not be opened" in page.inner_text(".json-notice") and "not in the public bucket" in page.inner_text(".json-notice")
    cataloged.problems.clear()  # the browser logs the expected 404 for that address


def test_without_a_catalog_results_are_discovered_from_summary_files(uncataloged):
    page = uncataloged.open()

    assert cataloged_summary(uncataloged) == "5 result sets in 2 sections"
    assert texts(page, ".section-heading h3") == ["Start here", "Paper figures", "Analyses"]
    assert texts(page, "#section-figures .entry-title") == ["Collection"]
    # Titles come from the analysis folder (or the analysis's own task), never the dataset folder.
    assert texts(page, "#section-analyses .entry-title") == [
        "Activity summary", "Chemical space", "Kinase probe analysis", "Quality",
    ]
    assert texts(page, "#section-analyses .entry-qualifier") == ["all", "compound_no_source7", "compound_no_source7"]

    page.fill("#search", "KCGS")
    uncataloged.settle()
    assert texts(page, ".claim-row") == ["49 n_in_collection.KCGS Kinase probe analysis: compound_no_source7"]


def cataloged_summary(session) -> str:
    return session.header()[2].split(" | ")[0]


def test_without_a_catalog_missing_and_foreign_paths_are_handled(uncataloged):
    # An envelope that is not named summary.json still gets its analysis header.
    page = uncataloged.open(f"?prefix={PROCESSED}chemical-space/all/")
    assert uncataloged.header()[:2] == ("ANALYSIS", "Chemical space")
    assert "1 declared output not in the public bucket" in page.inner_text(".summary-meta-row")

    # A declared file that is not in the bucket is labeled, not offered as a link that fails.
    page = uncataloged.open(f"?file={PROCESSED}chemical-space/all/chemical_space_summary.json")
    assert texts(page, ".manifest-actions") == ["Not in the public bucket"]
    assert page.locator(".manifest-actions a, .manifest-actions button").count() == 0

    # Paths written as absolute paths on another machine resolve to the same published files.
    page = uncataloged.open(f"?file={PROCESSED}kinase-probes/compound_no_source7/summary.json")
    assert texts(page, ".manifest-path") == ["kinase_heatmap.png", "probe_overlap.csv"]
    assert page.locator(".missing-note").count() == 0
    page.click(".manifest-item:nth-child(1) button")
    assert page.locator("#dialog-image").get_attribute("src") == (
        f"{BUCKET}/{PROCESSED}kinase-probes/compound_no_source7/kinase_heatmap.png"
    )


def test_without_a_catalog_headlines_are_only_top_level_results(uncataloged):
    # Top-level results in the analysis's own order: no parameter, nothing nested such as a byte size.
    page = uncataloged.open(f"?prefix={PROCESSED}activity-summary/")
    assert texts(page, ".summary-metric") == ["112,462 Analyzed compounds", "17,078 Phenotypically active compounds"]

    # An analysis whose top level holds only parameters shows no headline at all.
    page = uncataloged.open(f"?prefix={PROCESSED}kinase-probes/compound_no_source7/")
    assert uncataloged.header()[:2] == ("ANALYSIS", "Kinase probe analysis")
    assert page.locator(".summary-metric").count() == 0


def test_very_large_folders_are_listed_in_steps_and_drawn_in_batches(browser, tmp_path):
    """The publication prefix holds folders with tens of thousands of objects; opening one must stay cheap."""
    cache = tmp_path / "jump_production_null_cache/run/seed0"
    cache.mkdir(parents=True)
    for index in range(45):
        (cache / f"null_{index:04d}.npy").write_bytes(b"0")

    # Listing: the page reads five listing pages at a time (four objects each here) and says so.
    bucket = Bucket(tmp_path)
    session = Session(browser, bucket)
    page = session.open(f"?prefix={PUBLICATION}jump_production_null_cache/run/seed0/")
    assert page.locator(".file-row").count() == 20
    assert session.header()[2] == "0 folders | 20 files | more not yet listed"
    assert "more than the 20 entries listed so far" in page.inner_text(".listing-notice")
    listing_requests = len(bucket.requests)
    page.fill("#search", "null_0044")
    session.settle()
    assert page.locator(".file-row").count() == 0  # the filter only searches what is listed
    page.fill("#search", "")
    page.click(".listing-notice button:has-text('List more')")
    page.wait_for_function("document.querySelectorAll('.file-row').length === 40")
    page.click(".listing-notice button:has-text('List more')")
    page.wait_for_function("document.querySelectorAll('.file-row').length === 45")
    assert page.locator(".listing-notice").count() == 0
    assert session.header()[2] == "0 folders | 45 files"
    assert len(bucket.requests) - listing_requests == 7  # 5 pages, then 2: never the whole folder at once
    session.page.close()
    assert not session.problems

    # Drawing: a fully listed folder still shows its rows 200 at a time.
    wide = tmp_path / "wide/jump_production_null_cache/run/seed1"
    wide.mkdir(parents=True)
    for index in range(230):
        (wide / f"null_{index:04d}.npy").write_bytes(b"0")
    session = Session(browser, Bucket(tmp_path / "wide", page_size=1000))
    page = session.open(f"?prefix={PUBLICATION}jump_production_null_cache/run/seed1/")
    assert session.header()[2] == "0 folders | 230 files"
    assert page.locator(".file-row").count() == 200
    assert page.inner_text(".batch-more") == "Show 30 more files"
    page.click(".batch-more")
    assert page.locator(".file-row").count() == 230 and page.locator(".batch-more").count() == 0
    session.page.close()
    assert not session.problems


def test_newest_results_tree_with_a_catalog_is_the_one_shown(browser, outputs, tmp_path):
    """Publishing a new output tree beside the old one is one added line in RESULTS_PREFIXES."""
    publish(outputs, tmp_path, catalog=True)
    old_tree = tmp_path / "jump_production_datastore/processed"
    release = tmp_path / "jump_production_snapshots/release/data/processed"
    shutil.copytree(old_tree, release)
    (old_tree / "catalog.json").unlink()
    roots = "const RESULTS_PREFIXES = [`${DATASTORE_PREFIX}processed/`];"
    assert INDEX.read_text().count(roots) == 1
    index_text = INDEX.read_text().replace(
        roots,
        "const RESULTS_PREFIXES = [`${PUBLICATION_PREFIX}jump_production_snapshots/release/data/processed/`, "
        "`${DATASTORE_PREFIX}processed/`];",
    )
    release_prefix = f"{PUBLICATION}jump_production_snapshots/release/data/processed/"

    session = Session(browser, Bucket(tmp_path), index_text)
    page = session.open()
    assert session.header()[2] == "7 result sets in 3 sections | results from source revision 0123456"
    page.click(".entry-card:has-text('Kinase probe libraries')")
    session.settle()
    assert page.url == f"{ORIGIN}/index.html?prefix={release_prefix}kinase-probes/compound_no_source7/"
    assert page.locator(".image-card img").get_attribute("src") == (
        f"{BUCKET}/{release_prefix}kinase-probes/compound_no_source7/kinase_heatmap.png"
    )
    page = session.open("?q=17078")
    assert texts(page, ".claim-row") == ["17,078 phenotypically_active_compounds Phenotypic activity headline"]
    # The older tree stays browsable as files, and its envelopes still read correctly.
    page = session.open(f"?prefix={PROCESSED}activity-summary/")
    assert session.header()[:2] == ("ANALYSIS", "Activity summary")
    session.page.close()
    assert not session.problems

    # With no catalog anywhere, the last tree is read from its summary files.
    (release / "catalog.json").unlink()
    session = Session(browser, Bucket(tmp_path), index_text)
    page = session.open()
    assert session.header()[2].startswith("5 result sets in 2 sections")
    assert page.locator(".entry-card").first.get_attribute("href").startswith(f"/index.html?prefix={PROCESSED}")
    session.page.close()
    assert not session.problems


def test_opt_in_figure_is_presented_whether_or_not_the_catalog_lists_it(browser, outputs, tmp_path):
    # The default run does not build it: the catalog and the overview simply do not list it.
    published = tmp_path / "published"
    publish(outputs, published, catalog=True)
    session = Session(browser, Bucket(published))
    page = session.open()
    assert "Positive-control heatmap" not in texts(page, ".entry-title")

    # Built into a tree whose catalog predates it: no overview entry, but its folder is still an
    # analysis page with the legend, the figure and the table, described by its own summary.
    build_opt_in_figure(published / "jump_production_datastore/processed")
    session.page.close()
    session = Session(browser, Bucket(published))
    page = session.open(f"?prefix={PROCESSED}figures/figure_optional_heatmap/")
    assert session.header()[:2] == ("PAPER FIGURE", "Optional heatmap")
    assert "Positive-control reproducibility across sources" in page.inner_text(".summary-description")
    assert texts(page, ".summary-metric") == ["8 Compounds", "71 Matched pairs"]
    assert "Values per source." in page.locator(".section", has_text="Figure legends").inner_text()
    assert texts(page, ".image-name") == ["heatmap.png"]
    assert texts(page.locator(".file-row", has_text="heatmap.csv"), ".file-actions > *") == ["Open file", "Explore data"]
    assert "Positive-control heatmap" not in texts(session.open(), ".entry-title")
    session.page.close()
    assert not session.problems

    # Built before the catalog: listed under its section with the authored title and headline.
    rebuilt = tmp_path / "rebuilt-outputs"
    shutil.copytree(outputs, rebuilt)
    build_opt_in_figure(rebuilt / "data/processed")
    republished = tmp_path / "republished"
    publish(rebuilt, republished, catalog=True)
    session = Session(browser, Bucket(republished))
    page = session.open()
    assert texts(page, "#section-figures .entry-title") == ["Compound collection figure", "Positive-control heatmap"]
    page.click(".entry-card:has-text('Positive-control heatmap')")
    session.settle()
    assert session.header()[:2] == ("PAPER FIGURE", "Positive-control heatmap")
    assert texts(page, ".summary-metric") == ["71 Compound-source pairs"]
    session.page.close()
    assert not session.problems


@pytest.mark.skipif(os.environ.get("JUMP_BROWSER_LIVE") != "1", reason="set JUMP_BROWSER_LIVE=1 to read the published bucket")
def test_page_works_against_the_published_bucket(browser):
    session = Session(browser, None)
    page = session.open()
    page.wait_for_selector(".entry-card", timeout=120_000)
    count = int(re.match(r"(\d+) result sets", session.header()[2]).group(1))
    assert count >= 60 and page.locator(".entry-card").count() == count

    page.fill("#search", "activity summary")
    session.settle()
    page.wait_for_selector(".claim-row", timeout=60_000)
    page.click(".entry-card")
    session.settle()
    assert session.header()[0] == "ANALYSIS"
    assert page.locator(".summary-metric").count() >= 2

    page = session.open(f"?prefix={DATASTORE}raw/profiles/")
    row = page.locator(".file-row", has_text="compound_no_source7.parquet").first
    assert texts(row, ".file-actions > *") == ["Download", "Copy S3 path"]
    session.page.close()
    assert not session.problems
