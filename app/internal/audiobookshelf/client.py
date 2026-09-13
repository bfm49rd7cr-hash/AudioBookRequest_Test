from __future__ import annotations

import asyncio
import hashlib
import posixpath
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from aiohttp import ClientSession
from pydantic import BaseModel, TypeAdapter
from rapidfuzz import fuzz, process
from sqlmodel import Session

from app.internal.audiobookshelf.config import abs_config
from app.internal.audiobookshelf.types import (
    ABSBookItemMinified,
    ABSLibrary,
    ABSPodcastItem,
)
from app.internal.models import Audiobook
from app.util.connection import USER_AGENT
from app.util.db import get_session
from app.util.log import logger

# Keep the ABS snapshot fresh enough for availability checks without fetching the
# full library for every user search.
_ABS_LIBRARY_CACHE_TTL_SECONDS = 5 * 60
_ABS_LIBRARY_RETRY_BACKOFF_SECONDS = 30

# These are common edition/format qualifiers that may be appended to the title
# by one metadata source but omitted by another. They are only accepted when all
# additional title tokens are from this conservative list.
_TITLE_QUALIFIER_TOKENS = frozenset(
    {
        "a",
        "abridged",
        "audiobook",
        "complete",
        "and",
        "edition",
        "fassung",
        "gekurzt",
        "gekuerzt",
        "horbuch",
        "hoerbuch",
        "lesung",
        "novel",
        "roman",
        "unabridged",
        "uncut",
        "und",
        "ungekurzt",
        "ungekuerzt",
        "vollstandig",
        "vollstaendig",
        "vollstandige",
        "vollstaendige",
    }
)

_TITLE_QUALIFIER_PREFIXES = (
    "gekurzt",
    "gekuerzt",
    "horbuch",
    "hoerbuch",
    "ungekurzt",
    "ungekuerzt",
    "vollstandig",
    "vollstaendig",
)

_DISTINGUISHING_SEQUENCE_TOKENS = frozenset(
    {
        *(str(number) for number in range(1, 101)),
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eins",
        "zwei",
        "drei",
        "vier",
        "funf",
        "fuenf",
        "sechs",
        "sieben",
        "acht",
        "neun",
        "zehn",
    }
)


@dataclass(frozen=True, slots=True)
class _IndexedABSBook:
    item_id: str
    asin: str | None
    title_variants: tuple[str, ...]
    author_name: str
    author_name_lf: str
    narrator_name: str
    duration_seconds: float


@dataclass(slots=True)
class _ABSLibraryIndex:
    key: tuple[str, str, str]
    fetched_at: float
    items: list[_IndexedABSBook]
    by_asin: dict[str, _IndexedABSBook]
    by_title: dict[str, list[_IndexedABSBook]]
    fuzzy_titles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ABSMatch:
    item_id: str
    method: Literal["asin", "exact_metadata", "fuzzy_metadata"]
    title_score: float = 100.0
    author_score: float = 0.0
    narrator_score: float = 0.0
    duration_close: bool = False


_abs_library_index: _ABSLibraryIndex | None = None
_abs_library_index_lock = asyncio.Lock()
_abs_library_refresh_failure: tuple[tuple[str, str, str], float] | None = None


def _headers(session: Session) -> dict[str, str]:
    token = abs_config.get_api_token(session)
    assert token is not None
    return {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}


class _LibraryArray(BaseModel):
    libraries: list[ABSLibrary] = []


async def abs_get_libraries(
    session: Session, client_session: ClientSession
) -> list[ABSLibrary]:
    base_url = abs_config.get_base_url(session)
    if not base_url:
        return []
    url = posixpath.join(base_url, "api/libraries")
    try:
        async with client_session.get(url, headers=_headers(session)) as resp:
            if not resp.ok:
                logger.error(
                    "ABS: failed to fetch libraries",
                    status=resp.status,
                    reason=resp.reason,
                )
                return []
            data = _LibraryArray.model_validate(await resp.json())
            return data.libraries
    except Exception as e:
        logger.error("ABS: exception fetching libraries", error=str(e))
        return []


async def abs_trigger_scan(session: Session, client_session: ClientSession) -> bool:
    base_url = abs_config.get_base_url(session)
    lib_id = abs_config.get_library_id(session)
    if not base_url or not lib_id:
        return False
    url = posixpath.join(base_url, f"api/libraries/{lib_id}/scan")
    logger.debug("ABS: triggering library scan", library_id=lib_id, url=url)
    async with client_session.post(url, headers=_headers(session), json={}) as resp:
        if not resp.ok:
            logger.warning(
                "ABS: failed to trigger scan", status=resp.status, reason=resp.reason
            )
            return False
        return True


async def background_abs_trigger_scan():
    with next(get_session()) as session:
        async with ClientSession() as client_session:
            logger.debug("ABS: running background library scan trigger")
            success = await abs_trigger_scan(session, client_session)
            logger.info(
                "ABS: background library scan trigger complete", success=success
            )


class _ListResponseBook(BaseModel):
    results: list[ABSBookItemMinified] = []
    mediaType: Literal["book"]


class _ListResponsePodcast(BaseModel):
    results: list[ABSPodcastItem] = []
    mediaType: Literal["podcast"]


_ListResponse: TypeAdapter[_ListResponseBook | _ListResponsePodcast] = TypeAdapter(
    _ListResponseBook | _ListResponsePodcast
)


async def abs_list_library_items(
    session: Session,
    client_session: ClientSession,
    limit: int = 10,
) -> list[Audiobook]:
    """
    Fetch a page of items from the configured ABS library and map them to
    Audiobook objects to render on the homepage.
    """
    base_url = abs_config.get_base_url(session)
    lib_id = abs_config.get_library_id(session)
    if not base_url or not lib_id:
        return []

    url = posixpath.join(base_url, f"api/libraries/{lib_id}/items")
    params = {
        "limit": str(limit),
        "page": "0",
        "minified": "1",
        "sort": "addedAt",
        "desc": "1",
    }

    try:
        async with client_session.get(
            url, headers=_headers(session), params=params
        ) as resp:
            if not resp.ok:
                logger.debug(
                    "ABS: failed to list library items",
                    status=resp.status,
                    reason=resp.reason,
                )
                return []
            payload = _ListResponse.validate_python(await resp.json())
            if not isinstance(payload, _ListResponseBook):
                logger.warning(
                    "ABS: podcasts not supported in library listing", lib_id=lib_id
                )
                return []
    except Exception as e:
        logger.debug("ABS: exception listing library items", error=str(e))
        return []

    books: list[Audiobook] = []
    for item in payload.results:
        try:
            metadata = item.media.metadata
            title = metadata.title
            subtitle = metadata.subtitle
            authors = [metadata.authorName] if metadata.authorName else []
            narrators = [metadata.narratorName] if metadata.narratorName else []
            # Cover: ABS exposes cover via /api/items/:id/cover
            cover_image = posixpath.join(base_url, f"api/items/{item.id}/cover")
            # Duration in seconds -> minutes
            runtime_length_min = round((item.media.duration or 0.0) / 60)

            if metadata.publishedDate:
                try:
                    # Try ISO format
                    release_date = datetime.fromisoformat(
                        metadata.publishedDate.replace("Z", "+00:00")
                    )
                except Exception:
                    release_date = datetime.now()
            else:
                release_date = datetime.now()

            if not metadata.asin or not title:
                logger.warning(
                    "ABS: skipping library item with missing ASIN or title",
                    item_id=item.id,
                    asin=metadata.asin,
                    title=title,
                )
                continue

            book = Audiobook(
                asin=metadata.asin,
                title=title,
                subtitle=subtitle,
                authors=authors,
                narrators=narrators,
                cover_image=cover_image,
                release_date=release_date,
                runtime_length_min=runtime_length_min,
                downloaded=True,
            )
            books.append(book)
        except Exception as e:
            logger.debug("ABS: failed to map library item", error=str(e))

    return books


def _normalize_asin(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", value.strip().upper())


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""

    # casefold handles Unicode case rules better than lower(), including ß -> ss.
    normalized = value.casefold().translate(
        str.maketrans({"'": "", "’": "", "æ": "ae", "ø": "o", "ł": "l", "œ": "oe"})
    )
    # Remove combining marks so ö/o or é/e differences do not cause false misses.
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    # Preserve Unicode letters/numbers while treating punctuation as separators.
    normalized = "".join(char if char.isalnum() else " " for char in normalized)
    return " ".join(normalized.split())


def _title_variants(
    title: str | None,
    subtitle: str | None,
    alternate_title: str | None = None,
) -> tuple[str, ...]:
    variants: set[str] = set()
    normalized_title = _normalize_text(title)
    normalized_subtitle = _normalize_text(subtitle)
    normalized_alternate = _normalize_text(alternate_title)

    for base_title in (normalized_title, normalized_alternate):
        if not base_title:
            continue
        variants.add(base_title)
        if normalized_subtitle:
            # ABS and Audible do not always agree on whether a subtitle belongs
            # in the title or in the dedicated subtitle field.
            variants.add(f"{base_title} {normalized_subtitle}")

    return tuple(sorted(variants))


def _is_title_qualifier(token: str) -> bool:
    return token in _TITLE_QUALIFIER_TOKENS or token.startswith(
        _TITLE_QUALIFIER_PREFIXES
    )


def _strip_trailing_title_qualifiers(title: str) -> str:
    tokens = title.split()
    while len(tokens) > 1 and _is_title_qualifier(tokens[-1]):
        tokens.pop()
    return " ".join(tokens)


def _title_lookup_keys(title: str) -> tuple[str, ...]:
    stripped = _strip_trailing_title_qualifiers(title)
    return (title,) if stripped == title else (title, stripped)


def _titles_differ_only_by_qualifiers(left: str, right: str) -> bool:
    return left != right and (
        _strip_trailing_title_qualifiers(left) == right
        or _strip_trailing_title_qualifiers(right) == left
    )


def _sequence_tokens(title: str) -> frozenset[str]:
    return frozenset(
        token for token in title.split() if token in _DISTINGUISHING_SEQUENCE_TOKENS
    )


def _cache_key(session: Session) -> tuple[str, str, str] | None:
    base_url = abs_config.get_base_url(session)
    library_id = abs_config.get_library_id(session)
    token = abs_config.get_api_token(session)
    if not base_url or not library_id or not token:
        return None

    # A non-reversible token fingerprint prevents a permission/account change from
    # reusing a snapshot without retaining the secret itself in the cache key.
    token_fingerprint = hashlib.sha256(token.encode()).hexdigest()[:16]
    return base_url, library_id, token_fingerprint


async def _fetch_abs_library_index_items(
    session: Session,
    client_session: ClientSession,
) -> list[ABSBookItemMinified] | None:
    base_url = abs_config.get_base_url(session)
    library_id = abs_config.get_library_id(session)
    if not base_url or not library_id:
        return None

    url = posixpath.join(base_url, f"api/libraries/{library_id}/items")
    params = {
        # ABS documents limit=0 as "no limit". Requesting minified records keeps
        # the response substantially smaller than the expanded library model.
        "limit": "0",
        "page": "0",
        "minified": "1",
        "collapseseries": "0",
    }

    try:
        async with client_session.get(
            url, headers=_headers(session), params=params
        ) as resp:
            if not resp.ok:
                logger.warning(
                    "ABS: failed to refresh library index",
                    status=resp.status,
                    reason=resp.reason,
                )
                return None

            payload = _ListResponse.validate_python(await resp.json())
            if not isinstance(payload, _ListResponseBook):
                logger.warning(
                    "ABS: selected library is a podcast library",
                    library_id=library_id,
                )
                return []
            return payload.results
    except Exception as e:
        logger.warning("ABS: exception refreshing library index", error=str(e))
        return None


def _build_abs_library_index(
    key: tuple[str, str, str],
    items: list[ABSBookItemMinified],
) -> _ABSLibraryIndex:
    indexed_items: list[_IndexedABSBook] = []
    by_asin: dict[str, _IndexedABSBook] = {}
    by_title: dict[str, list[_IndexedABSBook]] = {}

    for item in items:
        metadata = item.media.metadata
        title_variants = _title_variants(
            metadata.title, metadata.subtitle, metadata.titleIgnorePrefix
        )
        normalized_asin = _normalize_asin(metadata.asin) or None
        if not title_variants and not normalized_asin:
            continue

        indexed = _IndexedABSBook(
            item_id=item.id,
            asin=normalized_asin,
            title_variants=title_variants,
            author_name=_normalize_text(metadata.authorName),
            author_name_lf=_normalize_text(metadata.authorNameLF),
            narrator_name=_normalize_text(metadata.narratorName),
            duration_seconds=item.media.duration or 0.0,
        )
        indexed_items.append(indexed)

        if normalized_asin:
            # Duplicate ASINs are not expected, but either matching item proves
            # that this Audible edition is already present.
            by_asin.setdefault(normalized_asin, indexed)

        for title in title_variants:
            for lookup_key in _title_lookup_keys(title):
                by_title.setdefault(lookup_key, []).append(indexed)

    return _ABSLibraryIndex(
        key=key,
        fetched_at=time.monotonic(),
        items=indexed_items,
        by_asin=by_asin,
        by_title=by_title,
        fuzzy_titles=tuple(by_title),
    )


async def _get_abs_library_index(
    session: Session,
    client_session: ClientSession,
) -> _ABSLibraryIndex | None:
    global _abs_library_index, _abs_library_refresh_failure

    key = _cache_key(session)
    if key is None:
        return None

    now = time.monotonic()
    cached = _abs_library_index
    if (
        cached is not None
        and cached.key == key
        and now - cached.fetched_at < _ABS_LIBRARY_CACHE_TTL_SECONDS
    ):
        return cached
    if (
        _abs_library_refresh_failure is not None
        and _abs_library_refresh_failure[0] == key
        and now - _abs_library_refresh_failure[1] < _ABS_LIBRARY_RETRY_BACKOFF_SECONDS
    ):
        return cached if cached is not None and cached.key == key else None

    # A lock prevents concurrent searches from all fetching the full ABS library
    # when the cache expires at the same time.
    async with _abs_library_index_lock:
        now = time.monotonic()
        cached = _abs_library_index
        if (
            cached is not None
            and cached.key == key
            and now - cached.fetched_at < _ABS_LIBRARY_CACHE_TTL_SECONDS
        ):
            return cached
        if (
            _abs_library_refresh_failure is not None
            and _abs_library_refresh_failure[0] == key
            and now - _abs_library_refresh_failure[1]
            < _ABS_LIBRARY_RETRY_BACKOFF_SECONDS
        ):
            return cached if cached is not None and cached.key == key else None

        items = await _fetch_abs_library_index_items(session, client_session)
        if items is None:
            _abs_library_refresh_failure = (key, now)
            # A stale known-good snapshot is safer than turning a temporary ABS
            # outage into duplicate request buttons for already-owned books.
            if cached is not None and cached.key == key:
                logger.warning(
                    "ABS: keeping stale library index after refresh failure",
                    cache_age_seconds=round(now - cached.fetched_at),
                    retry_after_seconds=_ABS_LIBRARY_RETRY_BACKOFF_SECONDS,
                )
                return cached
            return None

        _abs_library_refresh_failure = None
        _abs_library_index = _build_abs_library_index(key, items)
        logger.info(
            "ABS: library index refreshed",
            library_id=key[1],
            item_count=len(_abs_library_index.items),
            asin_count=len(_abs_library_index.by_asin),
        )
        return _abs_library_index


def _person_score(
    names: list[str],
    candidate_name: str,
    candidate_name_lf: str = "",
) -> float:
    if not names or (not candidate_name and not candidate_name_lf):
        return 0.0

    candidates = [name for name in (candidate_name, candidate_name_lf) if name]
    return max(
        (
            fuzz.token_sort_ratio(_normalize_text(name), candidate)
            for name in names
            for candidate in candidates
        ),
        default=0.0,
    )


def _duration_is_close(book: Audiobook, item: _IndexedABSBook) -> bool:
    if book.runtime_length_min <= 0 or item.duration_seconds <= 0:
        return False

    abs_minutes = item.duration_seconds / 60
    difference = abs(book.runtime_length_min - abs_minutes)
    # Allow small metadata/intro differences, but do not use duration as a
    # primary identifier.
    tolerance = max(10.0, book.runtime_length_min * 0.05)
    return difference <= tolerance


def _title_score(
    book_titles: tuple[str, ...],
    item_titles: tuple[str, ...],
) -> float:
    if not book_titles or not item_titles:
        return 0.0

    best_score = 0.0
    for book_title in book_titles:
        for item_title in item_titles:
            if book_title == item_title or _titles_differ_only_by_qualifiers(
                book_title, item_title
            ):
                return 100.0
            # Volume/part numbers distinguish adjacent books in the same series;
            # fuzzy similarity must never erase that difference.
            if _sequence_tokens(book_title) != _sequence_tokens(item_title):
                continue
            best_score = max(best_score, fuzz.ratio(book_title, item_title))

    return best_score


def _metadata_scores(
    book: Audiobook,
    item: _IndexedABSBook,
    book_titles: tuple[str, ...],
) -> tuple[float, float, float, bool]:
    return (
        _title_score(book_titles, item.title_variants),
        _person_score(book.authors, item.author_name, item.author_name_lf),
        _person_score(book.narrators, item.narrator_name),
        _duration_is_close(book, item),
    )


def _find_abs_match(
    index: _ABSLibraryIndex,
    book: Audiobook,
) -> _ABSMatch | None:
    normalized_asin = _normalize_asin(book.asin)

    # ASIN identifies the Audible edition. Once ABS metadata contains the same
    # ASIN, title formatting should not be able to invalidate that match.
    if normalized_asin and (item := index.by_asin.get(normalized_asin)):
        return _ABSMatch(item_id=item.item_id, method="asin")

    book_titles = _title_variants(book.title, book.subtitle)
    if not book_titles:
        return None

    # Exact normalized and qualifier-stripped titles are cheap indexed lookups.
    # Supporting metadata is still required so common titles do not collide.
    exact_candidates: dict[str, _IndexedABSBook] = {}
    for title in book_titles:
        for lookup_key in _title_lookup_keys(title):
            for item in index.by_title.get(lookup_key, []):
                exact_candidates[item.item_id] = item

    for item in exact_candidates.values():
        title_score, author_score, narrator_score, duration_close = _metadata_scores(
            book, item, book_titles
        )
        has_author_evidence = bool(book.authors) and bool(
            item.author_name or item.author_name_lf
        )
        if (has_author_evidence and author_score >= 90) or (
            not has_author_evidence and narrator_score >= 95 and duration_close
        ):
            return _ABSMatch(
                item_id=item.item_id,
                method="exact_metadata",
                title_score=title_score,
                author_score=author_score,
                narrator_score=narrator_score,
                duration_close=duration_close,
            )

    # RapidFuzz finds a bounded candidate set in native code. Detailed matching
    # then runs only for those candidates instead of scanning every library item
    # in Python for every Audible result.
    fuzzy_candidates: dict[str, _IndexedABSBook] = {}
    for book_title in book_titles:
        for matched_title, _, _ in process.extract(
            book_title,
            index.fuzzy_titles,
            scorer=fuzz.ratio,
            score_cutoff=88,
            limit=25,
        ):
            for item in index.by_title[matched_title]:
                fuzzy_candidates[item.item_id] = item

    best_scores = (0.0, 0.0, 0.0, False)
    for item in fuzzy_candidates.values():
        title_score, author_score, narrator_score, duration_close = _metadata_scores(
            book, item, book_titles
        )
        if title_score > best_scores[0]:
            best_scores = (
                title_score,
                author_score,
                narrator_score,
                duration_close,
            )

        if title_score >= 95 and author_score >= 92:
            return _ABSMatch(
                item_id=item.item_id,
                method="fuzzy_metadata",
                title_score=title_score,
                author_score=author_score,
                narrator_score=narrator_score,
                duration_close=duration_close,
            )

        if (
            title_score >= 90
            and author_score >= 95
            and narrator_score >= 92
            and duration_close
        ):
            return _ABSMatch(
                item_id=item.item_id,
                method="fuzzy_metadata",
                title_score=title_score,
                author_score=author_score,
                narrator_score=narrator_score,
                duration_close=duration_close,
            )

    logger.debug(
        "ABS: no confident library match",
        asin=book.asin,
        title=book.title,
        best_title_score=round(best_scores[0], 1),
        best_author_score=round(best_scores[1], 1),
        best_narrator_score=round(best_scores[2], 1),
        best_duration_close=best_scores[3],
    )
    return None


def _find_abs_matches(
    index: _ABSLibraryIndex, books: list[Audiobook]
) -> list[tuple[Audiobook, _ABSMatch]]:
    matches: list[tuple[Audiobook, _ABSMatch]] = []
    for book in books:
        if match := _find_abs_match(index, book):
            matches.append((book, match))
    return matches


async def abs_book_exists(
    session: Session,
    client_session: ClientSession,
    book: Audiobook,
) -> bool:
    """Check whether an Audible result is already present in the ABS library."""

    index = await _get_abs_library_index(session, client_session)
    if index is None:
        logger.debug(
            "ABS: library availability unknown because no index is available",
            asin=book.asin,
        )
        return False

    match = await asyncio.to_thread(_find_abs_match, index, book)
    if match is None:
        return False

    logger.debug(
        "ABS: book matched",
        asin=book.asin,
        abs_item_id=match.item_id,
        method=match.method,
        title_score=round(match.title_score, 1),
        author_score=round(match.author_score, 1),
        narrator_score=round(match.narrator_score, 1),
        duration_close=match.duration_close,
    )
    return True


async def abs_mark_downloaded_flags(
    session: Session,
    client_session: ClientSession,
    books: list[Audiobook],
) -> None:
    if not books or not abs_config.get_check_downloaded(session):
        return

    # Fetch/refresh the ABS snapshot once. All following checks are local and do
    # not generate one or two ABS API searches per Audible result.
    index = await _get_abs_library_index(session, client_session)
    if index is None:
        return

    to_match = [book for book in books if not book.downloaded]
    matches = await asyncio.to_thread(_find_abs_matches, index, to_match)
    for book, match in matches:
        logger.debug(
            "ABS: marking existing book as downloaded",
            asin=book.asin,
            abs_item_id=match.item_id,
            method=match.method,
            title_score=round(match.title_score, 1),
            author_score=round(match.author_score, 1),
            narrator_score=round(match.narrator_score, 1),
            duration_close=match.duration_close,
        )
        book.downloaded = True
        session.add(book)
