from typing import Literal

from pydantic import BaseModel


class ABSLibrary(BaseModel):
    id: str
    name: str
    mediaType: str


class ABSBookMinified(BaseModel):
    # Current ABS minified responses do not consistently include a media-level
    # id. The library-item id is the stable identifier used by ABR.
    id: str | None = None
    duration: float | None = None

    class _Metadata(BaseModel):
        title: str | None = None
        titleIgnorePrefix: str | None = None
        subtitle: str | None = None
        authorName: str | None = None
        authorNameLF: str | None = None
        narratorName: str | None = None
        publishedDate: str | None = None
        asin: str | None = None

    metadata: _Metadata


class ABSBook(BaseModel):
    id: str

    class _Metadata(BaseModel):
        title: str | None = None
        subtitle: str | None = None

        class _Author(BaseModel):
            id: str
            name: str

        authors: list[_Author]

    metadata: _Metadata


class ABSPodcast(BaseModel):
    pass


class ABSBookItem(BaseModel):
    id: str
    media: ABSBook
    mediaType: Literal["book"]


class ABSBookItemMinified(BaseModel):
    id: str
    media: ABSBookMinified
    mediaType: Literal["book"]


class ABSPodcastItem(BaseModel):
    id: str
    media: ABSPodcast
    mediaType: Literal["podcast"]


type ABSLibraryItem = ABSBookItemMinified | ABSPodcastItem
