"""Unit tests for domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models.document import Chunk, Document, DocumentMetadata, LegalDomain


@pytest.mark.unit
class TestDocumentMetadata:
    """Tests for DocumentMetadata validation."""

    def _make_meta(self, **kwargs: object) -> DocumentMetadata:
        defaults = {
            "source_file": "test.pdf",
            "title": "The Citizenship Act, 1955",
            "legal_domain": LegalDomain.CITIZENSHIP,
        }
        defaults.update(kwargs)
        return DocumentMetadata(**defaults)  # type: ignore[arg-type]

    def test_valid_metadata(self) -> None:
        meta = self._make_meta()
        assert meta.source_file == "test.pdf"
        assert meta.legal_domain == LegalDomain.CITIZENSHIP

    def test_empty_source_file_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make_meta(source_file="   ")

    def test_year_bounds(self) -> None:
        with pytest.raises(ValidationError):
            self._make_meta(year=1900)  # before 1947

        valid = self._make_meta(year=1955)
        assert valid.year == 1955

    def test_metadata_is_immutable(self) -> None:
        meta = self._make_meta()
        with pytest.raises(ValidationError):
            meta.title = "New Title"  # type: ignore[misc]


@pytest.mark.unit
class TestDocument:
    """Tests for Document model."""

    def _make_doc(self, content: str = "Sample legal text.") -> Document:
        meta = DocumentMetadata(
            source_file="test.pdf",
            title="Test Act",
        )
        return Document(content=content, metadata=meta)

    def test_document_auto_id(self) -> None:
        d = self._make_doc()
        assert d.id is not None
        assert len(d.id) == 36  # UUID4 format

    def test_word_count(self) -> None:
        d = self._make_doc("one two three four five")
        assert d.word_count == 5

    def test_unique_ids(self) -> None:
        d1 = self._make_doc()
        d2 = self._make_doc()
        assert d1.id != d2.id


@pytest.mark.unit
class TestChunk:
    """Tests for Chunk model."""

    def _make_chunk(self, start: int = 0, end: int = 100) -> Chunk:
        meta = DocumentMetadata(source_file="test.pdf", title="Test Act")
        return Chunk(
            document_id="doc-001",
            content="Some chunk content here.",
            metadata=meta,
            chunk_index=0,
            start_char=start,
            end_char=end,
        )

    def test_valid_chunk(self) -> None:
        chunk = self._make_chunk()
        assert chunk.char_length == 100

    def test_end_char_before_start_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._make_chunk(start=100, end=50)

    def test_chunk_auto_id(self) -> None:
        chunk = self._make_chunk()
        assert len(chunk.id) == 36
