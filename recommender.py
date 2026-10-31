"""
recommender.py - content-based filtering in pure Python (no numpy / scikit-learn).

How it works
    1. Every movie becomes a bag of tokens:  genres, keywords, director(s), language, decade.
    2. Tokens are weighted by field importance x IDF (rare tokens say more about taste).
    3. A user profile = weighted sum of the movies they rated (likes add, dislikes subtract)
       plus any genres they typed in.
    4. Every unseen movie is scored by cosine similarity to the profile.
    5. Optional filters (year, language, genres) and optional MMR re-ranking for variety.
    6. Every result comes with a human-readable explanation.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from catalog import Movie

# How much each kind of token matters. Tweak these and re-run evaluate.py to see the effect.
FIELD_WEIGHTS = {"g": 3.0, "k": 2.0, "d": 1.5, "l": 1.0, "e": 0.5}
FIELD_NAMES = {"g": "genre", "k": "theme", "d": "director", "l": "language", "e": "era"}

Vector = dict[str, float]


class RecommendationError(ValueError):
    """The request can't be answered (no taste information, unknown movie id...)."""


@dataclass
class Recommendation:
    movie: Movie
    score: float                                   # cosine similarity to the user's profile
    because: Optional[str] = None                  # title of the liked movie it most resembles
    shared: list[str] = field(default_factory=list)  # human-readable shared traits

    def reasons(self) -> str:
        parts = []
        if self.because:
            parts.append(f"Because you liked {self.because}")
        if self.shared:
            parts.append("shares " + ", ".join(self.shared))
        return " - ".join(parts) if parts else "Matches your taste"

    def to_dict(self) -> dict:
        d = self.movie.to_dict()
        d.update(score=round(self.score, 4), because=self.because, shared=self.shared,
                 reason=self.reasons())
        return d


def tokens_of(movie: Movie) -> list[str]:
    toks = [f"g:{g.lower()}" for g in movie.genres]
    toks += [f"k:{k.lower()}" for k in movie.keywords]
    toks += [f"d:{d.lower()}" for d in movie.directors]
    if movie.language:
        toks.append(f"l:{movie.language.lower()}")
    if movie.year:
        toks.append(f"e:{movie.year // 10 * 10}s")
    return toks


def dot(a: Vector, b: Vector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b[t] for t, w in a.items() if t in b)


def norm(v: Vector) -> float:
    return math.sqrt(sum(w * w for w in v.values()))


def unit(v: Vector) -> Vector:
    n = norm(v)
    return {t: w / n for t, w in v.items()} if n else {}


class Recommender:
    def __init__(self, movies: list[Movie], weights: Optional[dict[str, float]] = None):
        self.movies = movies
        self.by_id = {m.id: m for m in movies}
        self.weights = weights or FIELD_WEIGHTS
        df: Counter = Counter()
        self._tokens = {m.id: set(tokens_of(m)) for m in movies}
        for toks in self._tokens.values():
            df.update(toks)
        n = len(movies)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors: dict[int, Vector] = {
            m.id: unit({t: self.weights.get(t[0], 1.0) * self.idf[t] for t in self._tokens[m.id]})
            for m in movies}

    # ------------------------------------------------------------- profile
    def genre_vector(self, genres: Iterable[str]) -> Vector:
        vec = {f"g:{g.lower()}": self.weights["g"] * self.idf.get(f"g:{g.lower()}", 1.0) for g in genres}
        return unit(vec)

    def build_profile(self, ratings: Optional[dict[int, float]] = None,
                      genres: Iterable[str] = ()) -> Vector:
        """ratings: movie id -> 1..5 (4-5 like, 3 neutral, 1-2 dislike). genres: typed-in tastes."""
        profile: Vector = Counter()
        for mid, rating in (ratings or {}).items():
            if mid not in self.vectors:
                raise RecommendationError(f"Unknown movie id {mid}")
            weight = (rating - 3) / 2                       # 5 -> +1, 3 -> 0, 1 -> -1
            for t, w in self.vectors[mid].items():
                profile[t] += weight * w
        for t, w in self.genre_vector(genres).items():
            profile[t] += w
        profile = {t: w for t, w in profile.items() if w}
        if not profile or all(w <= 0 for w in profile.values()):
            raise RecommendationError("Tell me a genre you enjoy or a movie you liked first.")
        return unit(profile)

    # ----------------------------------------------------------- recommend
    def recommend(self, ratings: Optional[dict[int, float]] = None, genres: Iterable[str] = (),
                  n: int = 10, year_min: Optional[int] = None, year_max: Optional[int] = None,
                  languages: Iterable[str] = (), exclude_genres: Iterable[str] = (),
                  diverse: bool = False, diversity: float = 0.3) -> list[Recommendation]:
        ratings = ratings or {}
        profile = self.build_profile(ratings, genres)
        languages = {l.lower() for l in languages}
        banned = {g.lower() for g in exclude_genres}

        candidates = []
        for m in self.movies:
            if m.id in ratings:
                continue                                    # never recommend what they've rated
            if year_min and m.year < year_min or year_max and m.year > year_max:
                continue
            if languages and m.language.lower() not in languages:
                continue
            if banned and banned & {g.lower() for g in m.genres}:
                continue
            candidates.append((dot(profile, self.vectors[m.id]), m))
        candidates.sort(key=lambda sm: (-sm[0], sm[1].title))

        chosen = self._mmr(candidates, n, diversity) if diverse else candidates[:n]
        liked = [mid for mid, r in ratings.items() if r >= 4]
        return [self._explain(m, score, profile, liked) for score, m in chosen]

    def similar(self, movie_id: int, n: int = 10, **filters) -> list[Recommendation]:
        if movie_id not in self.by_id:
            raise RecommendationError(f"Unknown movie id {movie_id}")
        return self.recommend({movie_id: 5}, n=n, **filters)

    def similarity(self, a: int, b: int) -> float:
        return dot(self.vectors[a], self.vectors[b])

    # ------------------------------------------------------------- helpers
    def _mmr(self, candidates, n, diversity):
        """Maximal Marginal Relevance: trade a little relevance for a more varied list."""
        pool = candidates[:max(n * 5, 30)]
        picked: list[tuple[float, Movie]] = []
        while pool and len(picked) < n:
            def mmr(item):
                score, movie = item
                redundancy = max((self.similarity(movie.id, p.id) for _, p in picked), default=0.0)
                return (1 - diversity) * score - diversity * redundancy
            best = max(pool, key=mmr)
            pool.remove(best)
            picked.append(best)
        return picked

    def _explain(self, movie: Movie, score: float, profile: Vector, liked: list[int]) -> Recommendation:
        vec = self.vectors[movie.id]
        contributions = sorted(((profile.get(t, 0.0) * w, t) for t, w in vec.items()), reverse=True)
        shared = [self._pretty(t) for c, t in contributions
                  if c > 0 and not t.startswith("e:") and t != "l:english"][:3]
        because = None
        if liked:
            best = max(liked, key=lambda mid: self.similarity(movie.id, mid))
            if self.similarity(movie.id, best) > 0.05:
                because = self.by_id[best].title
        return Recommendation(movie, score, because, shared)

    @staticmethod
    def _pretty(token: str) -> str:
        kind, _, value = token.partition(":")
        if kind == "g":
            return value.title()                              # "sci-fi" -> "Sci-Fi"
        if kind == "d":
            return f"director {value.title()}"
        if kind == "l":
            return f"{value.title()} language"
        return value.replace("-", " ")                        # themes: "mind-bending" -> "mind bending"
