"""Raw fixtures preserve contradictory source text and stable bytes, without AI."""
import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
import unicodedata
from zipfile import ZipFile

from docx import Document
from PIL import Image
from pypdf import PdfReader
import pytest

from sr_adapter.relation_sources import read_source


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cross_format_fixtures.py"
SPEC = importlib.util.spec_from_file_location("cross_format_fixtures", MODULE_PATH)
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)


@pytest.fixture(scope="module")
def japanese_font():
    candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        # macOS filenames may be stored in decomposed Unicode.
        for spelling in (candidate, unicodedata.normalize("NFD", candidate)):
            if Path(spelling).is_file():
                return Path(spelling)
    pytest.skip("A Japanese-capable PNG font is required for the raw fixture test")


def _compact(text):
    return "".join(text.split())


def test_repeatable_sources_and_native_text_match_gold(tmp_path, japanese_font):
    first = fixtures.build_fixtures(tmp_path / "first", japanese_font)
    second = fixtures.build_fixtures(tmp_path / "second", japanese_font)
    assert len(first["sources"]) == len(second["sources"]) == 6
    assert {(s["profile"], s["format"]) for s in first["sources"]} == {
        (profile, format) for profile in ("A", "B") for format in ("pdf", "docx", "png")
    }
    assert [(s["source_name"], s["sha256"]) for s in first["sources"]] == [
        (s["source_name"], s["sha256"]) for s in second["sources"]
    ]
    for source in first["sources"]:
        data = Path(source["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
        assert len(data) == source["bytes"]
        expected = [source["title"]] + [unit["text"] for unit in source["text_units"]]
        assert len(source["text_units"]) == 5
        if source["format"] == "png":
            with Image.open(BytesIO(data)) as image:
                assert image.size == fixtures.IMAGE_SIZE
                assert image.mode == "RGB"
            continue
        reading = read_source(data, source["source_name"])
        text = "\n".join(unit["text"] for unit in reading["units"])
        assert _compact(text) == _compact("\n".join(expected))
        if source["format"] == "docx":
            assert [paragraph.text for paragraph in Document(BytesIO(data)).paragraphs] == expected
            with ZipFile(BytesIO(data)) as archive:
                assert archive.namelist() == sorted(archive.namelist())
                assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        else:
            pdf = PdfReader(BytesIO(data))
            assert len(pdf.pages) == 1
            assert {unit["locator"]["page"] for unit in reading["units"]} == {1}


def test_gold_disagreement_draft_and_missingness_are_explicit():
    profiles = fixtures.PROFILES
    assert profiles["A"]["expected"]["atlas_kind"] != profiles["B"]["expected"]["atlas_kind"]
    assert profiles["A"]["expected"]["reviewer"] == "審査部"
    assert profiles["B"]["expected"]["reviewer"] == "部長"
    assert profiles["B"]["expected"]["draft_qualifier"]
    assert {item["id"] for item in fixtures.EXPECTATIONS} >= {
        "same_name_nonidentity", "source_local_conditions", "conflict_no_winner",
        "draft_qualifier", "missing_final_approver", "evidence_survives_reuse",
    }
    for profile in profiles.values():
        text = "\n".join(profile["units"])
        assert all(value in text for value in ("LoanSystem L-01", "申請A", "2026-05-01", "2026-06-30", "100万円以下", "Atlas"))
        assert "優先資料は未決定" in text
        assert "最終承認者の氏名は記載されていない" in text
        assert profile["expected"]["final_approver_name"] is None
