"""Optional local OCR backends. Host code selects engines; no model/network calls."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from .knowledge import sha256

MAX_READING_BYTES = 4 * 1024 * 1024


class VisionOCR:
    """macOS on-device Vision OCR, with explicit language correction disabled."""
    def __init__(self, languages=None):
        if platform.system() != "Darwin" or not shutil.which("swift"):
            raise ValueError("Vision OCR requires macOS and the Swift command-line tools")
        self.languages = list(languages or ["ja-JP", "en-US"])
        self.script = Path(__file__).with_name("native") / "ocr_vision.swift"
        self.signature = {"engine": "apple-vision", "version": platform.mac_ver()[0],
            "languages": self.languages, "revision": 3, "language_correction": False,
            "implementation_sha256": sha256(self.script.read_bytes())}

    def __call__(self, data):
        with TemporaryDirectory(prefix="sr-vision-") as directory:
            source = Path(directory) / "frame.png"
            output = Path(directory) / "reading.json"
            source.write_bytes(data)
            with output.open("wb") as stream:
                subprocess.run([shutil.which("swift"), str(self.script), str(source), ",".join(self.languages)],
                    stdout=stream, stderr=subprocess.DEVNULL, check=True, timeout=90)
            with output.open("rb") as stream:
                raw = stream.read(MAX_READING_BYTES + 1)
        if len(raw) > MAX_READING_BYTES:
            raise ValueError("OCR reading exceeds 4 MiB")
        return json.loads(raw)


class TesseractOCR:
    """Portable Tesseract CLI; preserve low-confidence lines and word geometry."""
    def __init__(self, languages=None):
        self.command = shutil.which("tesseract")
        if self.command is None:
            raise ValueError("Install the Tesseract executable or select another OCR reader")
        version = subprocess.run([self.command, "--version"], capture_output=True, text=True,
            check=True, timeout=10).stdout.splitlines()[0]
        listing = subprocess.run([self.command, "--list-langs"], capture_output=True, text=True,
            check=True, timeout=10).stdout.splitlines()
        available = [line.strip() for line in listing[1:]]
        self.languages = list(languages or [v for v in ("jpn", "eng") if v in available])
        if not self.languages or not set(self.languages) <= set(available):
            raise ValueError("Requested OCR languages are not installed")
        directory = re.fullmatch(r'List of available languages in "(.+)" \([0-9]+\):', listing[0]) if listing else None
        if directory is None:
            raise ValueError("Tesseract must report its language-data directory for reproducible reuse")
        weights = {}
        for language in self.languages:
            digest, size = hashlib.sha256(), 0
            with (Path(directory[1]) / (language + ".traineddata")).open("rb") as stream:
                while chunk := stream.read(65536):
                    size += len(chunk)
                    if size > 128 * 1024 * 1024:
                        raise ValueError("OCR language data exceeds 128 MiB")
                    digest.update(chunk)
            weights[language] = digest.hexdigest()
        self.signature = {"engine": "tesseract", "version": version, "languages": self.languages,
            "traineddata_sha256": weights, "page_segmentation_mode": 3, "word_join": "single_space", "implementation_version": 2}

    def __call__(self, data):
        with TemporaryDirectory(prefix="sr-tesseract-") as directory:
            source, out = Path(directory) / "frame.png", Path(directory) / "result"
            source.write_bytes(data)
            subprocess.run([self.command, str(source), str(out), "-l", "+".join(self.languages), "--psm", "3", "tsv"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=60)
            with out.with_suffix(".tsv").open("rb") as stream:
                raw = stream.read(MAX_READING_BYTES + 1)
        if len(raw) > MAX_READING_BYTES:
            raise ValueError("OCR reading exceeds 4 MiB")
        return parse_tesseract(raw)


def parse_tesseract(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_READING_BYTES:
        raise ValueError("OCR reading exceeds 4 MiB")
    groups = {}
    # Tesseract emits literal text, not CSV-escaped fields. An unmatched quote
    # must never absorb the following word's TSV row into this word.
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row["level"] != "5" or not row["text"].strip():
            continue
        key = tuple(int(row[k]) for k in ("page_num", "block_num", "par_num", "line_num"))
        x, y, w, h = (int(row[k]) for k in ("left", "top", "width", "height"))
        confidence = float(row["conf"])
        groups.setdefault(key, []).append({"text": row["text"], "box": [x, y, x + w, y + h],
                                          "confidence": None if confidence < 0 else confidence / 100})
    lines = []
    for _, words in sorted(groups.items()):
        scores = [w["confidence"] for w in words if w["confidence"] is not None]
        lines.append({"text": " ".join(w["text"] for w in words),
            "box": [min(w["box"][0] for w in words), min(w["box"][1] for w in words),
                    max(w["box"][2] for w in words), max(w["box"][3] for w in words)],
            "confidence": sum(scores) / len(scores) if scores else None, "words": words})
    return {"lines": lines}


def local_reader(engine="auto", languages=None):
    if languages is not None and (not languages or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", v) for v in languages)):
        raise ValueError("Use explicit OCR language identifiers")
    if engine == "auto":
        engine = "vision" if platform.system() == "Darwin" and shutil.which("swift") else "tesseract"
    if engine == "vision":
        return VisionOCR(languages)
    if engine == "tesseract":
        return TesseractOCR(languages)
    raise ValueError("Use auto, vision or tesseract OCR engine")
