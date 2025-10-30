# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from sr_adapter.loaders import read_file_contents


def _create_png(path: Path, text: str) -> None:
    image = Image.new("RGB", (320, 120), color="white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 40), text, fill="black", font=font)

    info = PngImagePlugin.PngInfo()
    info.add_text("Description", text)
    image.save(str(path), pnginfo=info)


def test_xml_loader_returns_structure(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_text("<root><child attr='1'>value</child></root>", encoding="utf-8")

    text, meta = read_file_contents(xml, "application/xml")

    assert "value" in text
    assert meta["xml_valid"] is True
    assert meta["xml_root"] == "root"
    assert meta["xml_element_count"] >= 2


def test_rtf_loader_decodes_text(tmp_path: Path) -> None:
    rtf = tmp_path / "sample.rtf"
    rtf.write_text("{\\rtf1\\ansi This is \\'48\\'65\\'6c\\'6c\\'6f}", encoding="latin-1")

    text, meta = read_file_contents(rtf, "text/rtf")

    assert "Hello" in text
    assert meta["rtf_characters"] == len(text)


def test_eml_loader_extracts_plain_body(tmp_path: Path) -> None:
    eml = tmp_path / "sample.eml"
    boundary = "----=_Boundary123"
    eml.write_text(
        "\n".join(
            [
                "From: Example <sender@example.com>",
                "To: Reader <reader@example.com>",
                "Subject: Greetings",
                "MIME-Version: 1.0",
                f"Content-Type: multipart/alternative; boundary=\"{boundary}\"",
                "",
                f"--{boundary}",
                "Content-Type: text/plain; charset=utf-8",
                "",
                "Hello plain world!",
                f"--{boundary}",
                "Content-Type: text/html; charset=utf-8",
                "",
                "<p>Hello <strong>world</strong>!</p>",
                f"--{boundary}--",
                "",
            ]
        ),
        encoding="utf-8",
    )

    text, meta = read_file_contents(eml, "message/rfc822")

    assert "Hello plain world!" in text
    assert meta["email_parsed"] is True
    assert meta["email_attachment_count"] == 0
    assert meta["email_subject"] == "Greetings"


def test_ics_loader_counts_events(tmp_path: Path) -> None:
    ics = tmp_path / "schedule.ics"
    ics.write_text(
        "\n".join(
            [
                "BEGIN:VCALENDAR",
                "VERSION:2.0",
                "BEGIN:VEVENT",
                "SUMMARY:Planning Meeting",
                "DTSTART:20240101T090000",
                "DTEND:20240101T100000",
                "END:VEVENT",
                "END:VCALENDAR",
            ]
        ),
        encoding="utf-8",
    )

    text, meta = read_file_contents(ics, "text/calendar")

    assert "Planning Meeting" in text
    assert meta["ics_events"] == 1
    assert meta["ics_has_timezone"] is False


def test_yaml_loader_reports_documents(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join([
            "service:",
            "  host: localhost",
            "  retries: 3",
        ]),
        encoding="utf-8",
    )

    text, meta = read_file_contents(path, "text/yaml")

    assert "localhost" in text
    assert meta["yaml_documents"] == 1
    assert "service" in meta["yaml_root_keys"]


def test_toml_loader_parses_tables(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        "\n".join([
            "[service]",
            "host = \"localhost\"",
            "retries = 5",
        ]),
        encoding="utf-8",
    )

    text, meta = read_file_contents(path, "application/toml")

    assert "localhost" in text
    assert meta["toml_valid"] is True
    assert "service" in meta["toml_root_keys"]


def test_image_loader_extracts_text(tmp_path: Path) -> None:
    path = tmp_path / "invoice.png"
    _create_png(path, "Invoice #42 Total 19.99")

    text, meta = read_file_contents(path, "image/png")

    assert "Invoice #42" in text
    assert meta["image_has_text"] is True
    assert meta.get("image_text_sources")
    assert meta.get("image_mode") == "RGB"
