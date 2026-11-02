"""OCR boundaries, mixed-page routing and DB-only reuse; no native/model calls."""
import base64
import builtins
from copy import deepcopy
import io
import json
import sqlite3

from PIL import Image, ImageDraw
import pytest
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from sr_adapter.cli import main
from sr_adapter.local_ontology import inspect_ontology, read_ontology_frame, search_ontology
from sr_adapter.ocr_readers import parse_tesseract
from sr_adapter.ocr_sources import plan_ocr, render_ocr, verify_ocr_payloads
from sr_adapter.ontology_ingest import ingest_ontology_source, ontology_ingestion_status
from sr_adapter.tables import canonical_json
from test_ontology_ingest import proposed


def image():
    value = Image.new("RGB", (600, 240), "white")
    ImageDraw.Draw(value).text((20, 50), "Review requires approval.", fill="black")
    raw = io.BytesIO()
    value.save(raw, "PNG")
    return raw.getvalue()


def mixed_pdf(*, header=False, inline=False):
    raw = io.BytesIO()
    canvas = Canvas(raw, pagesize=(600, 400))
    canvas.drawString(30, 350, "Review requires approval.")
    canvas.showPage()
    if header:
        canvas.drawString(30, 350, "A long header must not hide the scanned table below.")
    if inline:
        canvas.drawInlineImage(Image.open(io.BytesIO(image())), 0, 0, width=600, height=240)
    else:
        canvas.drawImage(ImageReader(io.BytesIO(image())), 0, 0, width=600, height=240)
    canvas.showPage()
    canvas.save()
    return raw.getvalue()


class Reader:
    signature = {"engine": "fixture", "version": "1", "languages": ["eng"]}

    def __init__(self, reading=None):
        self.calls = []
        self.reading = reading or {"lines": [{"text": "Review requires approval.", "box": [5, 5, 200, 30],
                                            "confidence": 0.12, "words": []}]}

    def __call__(self, frame):
        self.calls.append(frame)
        return deepcopy(self.reading)


def source_package(data, name, **options):
    reader = Reader()
    plan = plan_ocr(data, name, reader_signature=reader.signature, **options)
    manifest, files = render_ocr(data, plan, reader)
    return manifest, {**files, "manifest.json": canonical_json(manifest)}, reader


@pytest.mark.parametrize("header,inline", [(False, False), (True, False), (True, True)])
def test_mixed_pdf_reads_only_scanned_page_and_preserves_low_confidence(header, inline):
    data = mixed_pdf(header=header, inline=inline)
    manifest, files, reader = source_package(data, "mixed.pdf")
    assert verify_ocr_payloads(files) == manifest
    assert len(reader.calls) == 1
    config = json.loads(files["ocr-config.json"])
    assert [p["method"] for p in config["pages"]] == ["embedded_text", "ocr"]
    units = json.loads(files["source.json"])["units"]
    assert [u["grounding"] for u in units] == ["source_text", "ocr_unverified"]
    assert units[1]["locator"]["confidence"] == 0.12
    assert units[1]["locator"]["box"] == [5, 5, 200, 30]
    assert units[1]["locator"]["page"] == 2
    assert files["source.bin"] == data


@pytest.mark.parametrize("extension,data", [("png", image()), ("pdf", mixed_pdf())])
def test_one_luna_request_keeps_pixels_and_reuses_without_ocr_or_render(extension, data, tmp_path, monkeypatch):
    reader, calls = Reader(), []
    source, db = tmp_path / ("source." + extension), tmp_path / "ontology.sqlite"
    source.write_bytes(data)
    def complete(wire):
        calls.append(wire)
        return proposed(wire)
    options = dict(ocr="auto", ocr_reader=reader, complete=complete, adapter_id="fixture", execute=True)
    result = ingest_ontology_source(source, db, **options)
    assert result["status"] == "complete", result
    assert calls[0]["model"] == "gpt-6-luna"
    assert len(calls) == result["submitted_requests"] == len(reader.calls) == 1
    expected_pages = 1 if extension == "png" else 2
    parts = calls[0]["input"][1]["content"]
    images = [base64.b64decode(p["image_url"].split(",", 1)[1]) for p in parts if p["type"] == "input_image"]
    assert len(images) == expected_pages
    import sr_adapter.ocr_sources as ocr_module
    monkeypatch.setattr(ocr_module, "_render", lambda *a: pytest.fail("Unchanged source rendered again"))
    reused = ingest_ontology_source(source, db, **options)
    assert reused["outcome"] == "already_active"
    assert reused["submitted_requests_this_call"] == reused["ocr_pages_this_call"] == reused["rendered_pages_this_call"] == 0
    with pytest.raises(ValueError, match="Entry changed"):
        ingest_ontology_source(source, db, **{**options, "ocr_dpi": 200})
    assert len(reader.calls) == len(calls) == 1
    source.unlink()
    original_import = builtins.__import__
    def offline_import(name, *args, **kwargs):
        if name.startswith(("pyarrow", "pypdfium2", "sr_adapter.ocr")):
            pytest.fail("DB retrieval imported an extraction dependency: " + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", offline_import)
    hit = search_ontology(db, kind="claim")["hits"][0]
    trace = inspect_ontology(db, **hit["reference"])["value"]
    package_id = trace["claims"][0]["package_id"]
    for number, pixels in enumerate(images, 1):
        saved = read_ontology_frame(db, collection="default", name=source.name,
            snapshot_id=result["snapshot_id"], package_id=package_id, page=number)
        assert saved["data"] == pixels
        assert saved["reading_status"] == "unverified"
    with pytest.raises(ValueError, match="absent"):
        read_ontology_frame(db, collection="default", name=source.name,
            snapshot_id=result["snapshot_id"], package_id=package_id, page=3)
    status = ontology_ingestion_status(db, result["run_id"])
    assert {f["path"] for f in status["stored_files"]} >= {"source.bin", "ocr/frame-0001.png", "ocr/page-0001.json", "semantics-response.json"}


def test_ocr_failure_keeps_frames_and_invalid_reading_without_model_or_activation(tmp_path):
    source, db = tmp_path / "source.pdf", tmp_path / "ontology.sqlite"
    source.write_bytes(mixed_pdf())
    reader = Reader({"lines": [{"text": "Bad geometry", "box": [-1, 0, 40, 20], "confidence": None, "words": []}]})
    result = ingest_ontology_source(source, db, ocr="auto", ocr_reader=reader, complete=lambda w: pytest.fail("Model called"),
        adapter_id="fixture", execute=True)
    assert result["status"] == "failed" and result["submitted_requests"] == 0
    status = ontology_ingestion_status(db, result["run_id"])
    assert {f["path"] for f in status["stored_files"]} >= {"source.bin", "ocr/frame-0002.png", "ocr/page-0002.json"}
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM sro_entries").fetchone()[0] == 0


def test_selection_always_mode_and_cli_preparation_without_model(tmp_path, monkeypatch, capsys):
    reader, source, db = Reader(), tmp_path / "source.pdf", tmp_path / "ontology.sqlite"
    source.write_bytes(mixed_pdf())
    monkeypatch.setattr("sr_adapter.ocr_readers.local_reader", lambda *args: reader)
    assert main(["knowledge", "ontology", "ingest", str(source), "--database", str(db), "--ocr", "always", "--pages", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model"] == "gpt-6-luna" and result["submitted_requests"] == 0
    assert result["status"] == "prepared" and len(reader.calls) == 1
    assert any("selected physical pages" in note for note in result["warnings"])


@pytest.mark.parametrize("options", [{"pages": [2, 1]}, {"pages": [1, 1]}, {"pages": [3]}, {"pages": [True]},
    {"pages": []}, {"dpi": 301}, {"dpi": True}, {"mode": "fallback"}])
def test_invalid_reading_options_do_not_reach_ocr(options):
    with pytest.raises(ValueError):
        plan_ocr(mixed_pdf(), "source.pdf", reader_signature=Reader.signature, **options)


@pytest.mark.parametrize("change", ["page_count", "routing", "hash", "geometry", "confidence", "word", "missing_frame"])
def test_saved_ocr_tampering_rejected(change):
    _, files, _ = source_package(image(), "image.png")
    plan, page = json.loads(files["ocr-config.json"]), json.loads(files["page-0001.json"])
    if change == "page_count":
        plan["page_count"] = 2
    elif change == "routing":
        plan["pages"][0]["method"] = "embedded_text"
    elif change == "hash":
        page["frame_sha256"] = "0" * 64
    elif change == "geometry":
        page["reading"]["lines"][0]["box"] = [0, 0, 10000, 20]
    elif change == "confidence":
        page["reading"]["lines"][0]["confidence"] = float("inf")
    elif change == "word":
        page["reading"]["lines"][0]["words"] = [{"text": "wrong", "box": [0, 0, 2, 2], "confidence": None}]
    else:
        files.pop("frame-0001.png")
    files["ocr-config.json"], files["page-0001.json"] = canonical_json(plan), json.dumps(page).encode()
    with pytest.raises((ValueError, KeyError)):
        verify_ocr_payloads(files)


def test_empty_ocr_is_explicit_and_tesseract_keeps_word_boxes_and_low_scores():
    reader = Reader({"lines": []})
    plan = plan_ocr(image(), "image.png", reader_signature=reader.signature)
    manifest, files = render_ocr(image(), plan, reader)
    assert any("not proof" in note for note in manifest["warnings"])
    assert json.loads(files["source.json"])["units"] == []
    raw = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t5\t10\t20\t10\t12\tReview\n"
        "5\t1\t1\t1\t1\t2\t30\t10\t30\t10\t-1\tapproval\n").encode()
    line = parse_tesseract(raw)["lines"][0]
    assert line["text"] == "Review approval" and line["confidence"] == 0.12
    assert line["box"] == [5, 10, 60, 20]
    assert line["words"][1]["confidence"] is None


def test_exif_orientation_preserves_original_and_binds_boxes_to_rotated_pixels():
    original = Image.new("RGB", (320, 160), "white")
    exif = original.getexif()
    exif[274] = 6
    raw = io.BytesIO()
    original.save(raw, "JPEG", exif=exif)
    reader = Reader({"lines": [{"text": "Rotated", "box": [5, 5, 100, 30], "confidence": None, "words": []}]})
    data = raw.getvalue()
    plan = plan_ocr(data, "rotated.jpg", reader_signature=reader.signature)
    manifest, files = render_ocr(data, plan, reader)
    assert files["source.bin"] == data
    page = json.loads(files["page-0001.json"])
    assert (page["width"], page["height"]) == (160, 320)
    assert verify_ocr_payloads({**files, "manifest.json": canonical_json(manifest)}) == manifest


def test_ocr_v4_compacts_model_context_but_keeps_full_bindings_and_old_schema(tmp_path):
    from sr_adapter.relation_dataset_store import _save_snapshot, export_relation_dataset, _load_dataset
    from sr_adapter.semantic_proposals import _source, proposal_context, proposal_request, proposal_schema
    manifest, files, _ = source_package(image(), "source.png")
    _save_snapshot(tmp_path / "source", {k: v for k, v in files.items() if k != "manifest.json"}, manifest)
    export_relation_dataset([tmp_path / "source"], tmp_path / "dataset", profile="narrative")
    inputs = _load_dataset(tmp_path / "dataset", allow_view=False)[1]
    dataset = _source(inputs)
    old, _, _ = proposal_context(dataset, version=3)
    compact, _, bindings = proposal_context(dataset, version=4)
    assert len(canonical_json(compact)) < len(canonical_json(old)) / 2
    unit = next(r for r in compact["records"] if r["table"] == "evidence")
    assert unit["locations"]["/confidence"] == 0.12 and unit["locations"]["/box/0"] == 5
    assert "source_sha256" not in unit["record"]
    assert "source_sha256" in bindings[unit["id"]]["record"]
    assert all(r["table"] != "locations" for r in compact["records"])
    v4 = proposal_schema(version=4)["$defs"]["ProposedOverlayV2"]
    assert v4["properties"]["terms"]["minItems"] == 1
    assert "minItems" not in proposal_schema(version=3)["$defs"]["ProposedOverlayV2"]["properties"]["terms"]
    assert proposal_request(inputs, model="fixture", version=4)["reasoning"] == {"effort": "medium"}


def test_db_page_tampering_is_detected_without_rerunning_ocr(tmp_path):
    source, db = tmp_path / "source.png", tmp_path / "onto.db"
    source.write_bytes(image())
    result = ingest_ontology_source(source, db, ocr="auto", ocr_reader=Reader(), complete=proposed, adapter_id="fixture", execute=True)
    assert result["status"] == "complete"
    hit = search_ontology(db, kind="claim")["hits"][0]
    package_id = inspect_ontology(db, **hit["reference"])["value"]["claims"][0]["package_id"]
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE sro_files SET data=? WHERE path LIKE ?", (b"corrupt", "%/frame-0001.png"))
    with pytest.raises(ValueError, match="absent or oversized|differs"):
        read_ontology_frame(db, collection="default", name=source.name, snapshot_id=result["snapshot_id"], package_id=package_id, page=1)


def test_tesseract_reuse_signature_changes_when_language_weights_change(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sr_adapter.ocr_readers import TesseractOCR
    weights = tmp_path / "eng.traineddata"
    weights.write_bytes(b"first weights")
    monkeypatch.setattr("sr_adapter.ocr_readers.shutil.which", lambda *args: "fixture-tesseract")
    def run(args, **kwargs):
        text = "tesseract 5.5.0\n" if args[-1] == "--version" else f'List of available languages in "{tmp_path}" (1):\neng\n'
        return SimpleNamespace(stdout=text)
    monkeypatch.setattr("sr_adapter.ocr_readers.subprocess.run", run)
    first = TesseractOCR().signature
    weights.write_bytes(b"changed weights")
    assert first != TesseractOCR().signature
    with pytest.raises(ValueError, match="not installed"):
        TesseractOCR(["jpn"])
