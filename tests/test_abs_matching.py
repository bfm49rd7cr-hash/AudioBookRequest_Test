import unittest
from datetime import UTC, datetime

from app.internal.audiobookshelf import client as abs_client
from app.internal.audiobookshelf.client import (
    _ABSLibraryIndex,
    _find_abs_match,
    _IndexedABSBook,
    _normalize_text,
    _title_variants,
)
from app.internal.models import Audiobook


def make_book(
    *,
    asin: str = "B000000000",
    title: str,
    subtitle: str | None = None,
    authors: list[str] | None = None,
    narrators: list[str] | None = None,
    runtime_length_min: int = 600,
) -> Audiobook:
    return Audiobook(
        asin=asin,
        title=title,
        subtitle=subtitle,
        authors=authors or [],
        narrators=narrators or [],
        cover_image=None,
        release_date=datetime(2020, 1, 1, tzinfo=UTC),
        runtime_length_min=runtime_length_min,
    )


def make_index(*items: _IndexedABSBook) -> _ABSLibraryIndex:
    by_asin = {item.asin: item for item in items if item.asin}
    by_title: dict[str, list[_IndexedABSBook]] = {}
    for item in items:
        for title in item.title_variants:
            by_title.setdefault(title, []).append(item)
    return _ABSLibraryIndex(
        key=("https://abs.example", "library", 1),
        fetched_at=0,
        items=list(items),
        by_asin=by_asin,
        by_title=by_title,
    )


def make_abs_item(
    *,
    item_id: str = "abs-item",
    asin: str | None = None,
    title: str,
    subtitle: str | None = None,
    author: str = "",
    author_lf: str = "",
    narrator: str = "",
    duration_seconds: float = 0,
) -> _IndexedABSBook:
    return _IndexedABSBook(
        item_id=item_id,
        asin=asin,
        title_variants=_title_variants(title, subtitle),
        author_name=_normalize_text(author),
        author_name_lf=_normalize_text(author_lf),
        narrator_name=_normalize_text(narrator),
        duration_seconds=duration_seconds,
    )


class ABSMatchingTests(unittest.TestCase):
    def test_exact_asin_wins_over_metadata_differences(self):
        index = make_index(
            make_abs_item(
                asin="B012345678",
                title="The Hobbit (Unabridged)",
                author="J. R. R. Tolkien",
            )
        )
        match = _find_abs_match(
            index,
            make_book(
                asin="b012345678",
                title="Different Metadata Entirely",
                authors=["Another Author"],
            ),
        )
        assert match is not None
        self.assertEqual(match.method, "asin")

    def test_subtitle_and_author_order_match(self):
        index = make_index(
            make_abs_item(
                title="Project Hail Mary",
                subtitle="A Novel",
                author="King, Stephen",
            )
        )
        match = _find_abs_match(
            index,
            make_book(
                title="Project Hail Mary",
                subtitle="A Novel",
                authors=["Stephen King"],
            ),
        )
        assert match is not None
        self.assertEqual(match.method, "exact_metadata")

    def test_unicode_and_harmless_edition_qualifier_match(self):
        index = make_index(
            make_abs_item(
                title="Der Hobbit – Ungekürzte Lesung",
                author="J. R. R. Tolkien",
            )
        )
        match = _find_abs_match(
            index,
            make_book(title="Der Hobbit", authors=["J. R. R. Tolkien"]),
        )
        assert match is not None
        self.assertEqual(match.method, "exact_metadata")
        self.assertEqual(_normalize_text("Straße"), "strasse")

    def test_cache_invalidation_clears_snapshot(self):
        abs_client._abs_library_index = make_index(
            make_abs_item(title="Cached Book", author="Author")
        )
        abs_client._invalidate_abs_library_index()
        self.assertIsNone(abs_client._abs_library_index)

    def test_similar_series_title_does_not_match(self):
        index = make_index(make_abs_item(title="Dune Messiah", author="Frank Herbert"))
        self.assertIsNone(
            _find_abs_match(index, make_book(title="Dune", authors=["Frank Herbert"]))
        )

    def test_same_title_different_author_does_not_match(self):
        index = make_index(make_abs_item(title="The Outsider", author="Albert Camus"))
        self.assertIsNone(
            _find_abs_match(
                index, make_book(title="The Outsider", authors=["Stephen King"])
            )
        )


if __name__ == "__main__":
    unittest.main()
