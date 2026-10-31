"""
catalog.py - the movie catalog and forgiving title / genre lookup.

Movies live in data/movies.csv (id, title, year, genres, director, language, keywords).
Add rows to that file to grow the catalog - no code changes needed.
"""
from __future__ import annotations

import csv
import difflib
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "movies.csv"


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    year: int
    genres: tuple[str, ...]
    director: str = ""
    language: str = ""
    keywords: tuple[str, ...] = ()

    @property
    def directors(self) -> list[str]:
        return [d.strip() for d in self.director.split(" & ") if d.strip()]

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "year": self.year, "genres": list(self.genres),
                "director": self.director, "language": self.language, "keywords": list(self.keywords)}


def load_catalog(path: Path | str = DEFAULT_CSV) -> list[Movie]:
    movies = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            movies.append(Movie(
                id=int(row["id"]), title=row["title"].strip(), year=int(row["year"]),
                genres=tuple(g for g in row["genres"].split("|") if g),
                director=row.get("director", "").strip(), language=row.get("language", "").strip(),
                keywords=tuple(k for k in row.get("keywords", "").split("|") if k)))
    if not movies:
        raise ValueError(f"No movies found in {path}")
    return movies


# --------------------------------------------------------------------------
# Lookup helpers
# --------------------------------------------------------------------------
GENRE_ALIASES = {
    "scifi": "Sci-Fi", "sci fi": "Sci-Fi", "science fiction": "Sci-Fi", "sf": "Sci-Fi",
    "kids": "Family", "cartoon": "Animation", "animated": "Animation", "scary": "Horror",
    "romcom": "Romance", "love": "Romance", "doc": "Biography", "true story": "Biography",
    "historical": "History", "sports": "Sport", "musical": "Musical", "music": "Music",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("&", "and")).strip()


class Lookup:
    """Fuzzy search over titles and genres (typo and partial-name friendly)."""

    def __init__(self, movies: list[Movie]):
        self.movies = movies
        self.genres = sorted({g for m in movies for g in m.genres})
        self.languages = sorted({m.language for m in movies if m.language})
        self._by_title = {_norm(m.title): m for m in movies}

    def find(self, query: str, limit: int = 5) -> list[Movie]:
        """Best matches first: exact, then prefix/substring, then fuzzy."""
        q = _norm(query)
        if not q:
            return []
        if q in self._by_title:
            return [self._by_title[q]]
        found: list[Movie] = []
        for key, movie in self._by_title.items():
            if key.startswith(q):
                found.append(movie)
        for key, movie in self._by_title.items():
            if q in key and movie not in found:
                found.append(movie)
        for key in difflib.get_close_matches(q, list(self._by_title), n=limit, cutoff=0.6):
            if self._by_title[key] not in found:
                found.append(self._by_title[key])
        return found[:limit]

    def genre(self, text: str) -> str | None:
        q = _norm(text)
        if not q:
            return None
        table = {_norm(g): g for g in self.genres}
        table.update({_norm(a): g for a, g in GENRE_ALIASES.items() if g in self.genres})
        if q in table:
            return table[q]
        close = difflib.get_close_matches(q, list(table), n=1, cutoff=0.75)
        return table[close[0]] if close else None

    def language(self, text: str) -> str | None:
        table = {l.lower(): l for l in self.languages}
        q = text.strip().lower()
        if q in table:
            return table[q]
        close = difflib.get_close_matches(q, list(table), n=1, cutoff=0.75)
        return table[close[0]] if close else None
