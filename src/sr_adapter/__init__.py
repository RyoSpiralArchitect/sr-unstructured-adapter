"""Tools for transforming unstructured files into normalized block documents."""

from .pipeline import convert, convert_many
from .schema import Block, Document, Span
from .sniff import detect_type

__all__ = ["Block", "Document", "Span", "convert", "convert_many", "detect_type"]
