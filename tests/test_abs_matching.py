import unittest
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession
from sqlmodel import Session

from app.internal.audiobookshelf import client as abs_client
from app.internal.audiobookshelf.client import (
    _ABSLibraryIndex,
    _find_abs_match,
    _IndexedABSBook,
    _ListResponse,
    _ListResponseBook,
    _normalize_text,
    _title_lookup_keys,
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
            for lookup_key in _title_lookup_keys(title):
                by_title.setdefault(lookup_key, []).append(item)
    return _ABSLibraryIndex(
        key=("https://abs.example", "library", "token-fingerprint"),
        fetched_at=0,
        items=list(items),
        by_asin=by_asin,
        by_title=by_title,
        fuzzy_titles=tuple(by_title),
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

    def test_adjacent_volume_numbers_do_not_match(self):
        index = make_index(
            make_abs_item(
                title="Chronicles of the Starship Volume 2", author="Same Author"
            )
        )
        self.assertIsNone(
            _find_abs_match(
                index,
                make_book(
                    title="Chronicles of the Starship Volume 1",
                    authors=["Same Author"],
                ),
            )
        )

    def test_reordered_title_words_do_not_match(self):
        index = make_index(make_abs_item(title="Man Bites Dog", author="Same Author"))
        self.assertIsNone(
            _find_abs_match(
                index, make_book(title="Dog Bites Man", authors=["Same Author"])
            )
        )

    def test_partial_author_name_is_not_supporting_evidence(self):
        index = make_index(make_abs_item(title="Shared Title", author="John"))
        self.assertIsNone(
            _find_abs_match(
                index, make_book(title="Shared Title", authors=["John Smith"])
            )
        )

    def test_duration_alone_does_not_prove_same_title(self):
        index = make_index(
            make_abs_item(title="Shared Title", duration_seconds=600 * 60)
        )
        self.assertIsNone(
            _find_abs_match(
                index, make_book(title="Shared Title", runtime_length_min=600)
            )
        )

    def test_small_typo_with_same_author_still_matches(self):
        index = make_index(
            make_abs_item(title="Project Hail Marry", author="Andy Weir")
        )
        match = _find_abs_match(
            index, make_book(title="Project Hail Mary", authors=["Andy Weir"])
        )
        assert match is not None
        self.assertEqual(match.method, "fuzzy_metadata")

    def test_unicode_name_punctuation_and_compatibility(self):
        self.assertEqual(_normalize_text("O’Connor"), _normalize_text("OConnor"))
        self.assertEqual(_normalize_text("Łódź"), _normalize_text("Lodz"))
        self.assertEqual(_normalize_text("Æsir"), _normalize_text("Aesir"))


class ABSCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_library_fetch_uses_minified_unlimited_request(self):
        class FakeResponse:
            ok = True
            status = 200
            reason = "OK"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self):
                return {
                    "mediaType": "book",
                    "results": [
                        {
                            "id": "item-1",
                            "mediaType": "book",
                            "media": {
                                "duration": None,
                                "metadata": {
                                    "title": "A Book",
                                    "asin": "B012345678",
                                },
                            },
                        }
                    ],
                }

        class FakeClient:
            url = ""
            params: dict[str, str] = {}

            def get(self, url, *, headers, params):
                self.url = url
                self.params = params
                return FakeResponse()

        client = FakeClient()
        client_session = cast(ClientSession, client)
        session = cast(Session, object())
        with (
            patch.object(
                abs_client.abs_config,
                "get_base_url",
                return_value="https://abs.example/",
            ),
            patch.object(
                abs_client.abs_config, "get_library_id", return_value="library-1"
            ),
            patch.object(abs_client.abs_config, "get_api_token", return_value="token"),
        ):
            items = await abs_client._fetch_abs_library_index_items(
                session, client_session
            )

        assert items is not None
        self.assertEqual(len(items), 1)
        self.assertEqual(
            client.url, "https://abs.example/api/libraries/library-1/items"
        )
        self.assertEqual(
            client.params,
            {
                "limit": "0",
                "page": "0",
                "minified": "1",
                "collapseseries": "0",
            },
        )

    async def test_empty_book_list_does_not_fetch_library(self):
        session = cast(Session, object())
        client_session = cast(ClientSession, object())
        await abs_client.abs_mark_downloaded_flags(session, client_session, [])

    def test_minified_response_allows_missing_optional_media_fields(self):
        payload = _ListResponse.validate_python(
            {
                "mediaType": "book",
                "results": [
                    {
                        "id": "item-1",
                        "mediaType": "book",
                        "media": {
                            "metadata": {
                                "title": "A Book",
                                "asin": "B012345678",
                            }
                        },
                    }
                ],
            }
        )
        self.assertIsInstance(payload, _ListResponseBook)
        assert isinstance(payload, _ListResponseBook)
        self.assertIsNone(payload.results[0].media.duration)
        self.assertIsNone(payload.results[0].media.metadata.authorName)

    async def test_failed_refresh_uses_backoff_before_retrying(self):
        cached = make_index(make_abs_item(title="Cached Book", author="Author"))
        cached.fetched_at = 0
        abs_client._abs_library_index = cached
        abs_client._abs_library_refresh_failure = None
        fetch = AsyncMock(return_value=None)

        session = cast(Session, object())
        client_session = cast(ClientSession, object())
        with (
            patch.object(abs_client, "_cache_key", return_value=cached.key),
            patch.object(abs_client, "_fetch_abs_library_index_items", fetch),
            patch.object(abs_client.time, "monotonic", side_effect=[1000, 1000, 1001]),
        ):
            first = await abs_client._get_abs_library_index(session, client_session)
            second = await abs_client._get_abs_library_index(session, client_session)

        self.assertIs(first, cached)
        self.assertIs(second, cached)
        self.assertEqual(fetch.await_count, 1)
        abs_client._abs_library_index = None
        abs_client._abs_library_refresh_failure = None


if __name__ == "__main__":
    unittest.main()
