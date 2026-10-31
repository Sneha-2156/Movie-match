"""
evaluate.py - how good are the recommendations?

Two ways to measure quality:

 1. PERSONA BENCHMARK (works offline, built in)
    12 hand-written taste personas, each with 8-10 movies they love. For each persona we reveal
    only 3 of them (a "cold start"), build the profile from those, and check whether the
    recommender finds the remaining hidden favourites in its top-K. Repeated over many random draws.
    -> A sanity check of the whole pipeline. The personas are written by hand, so treat the
       numbers as a relative comparison between models, not as real-world accuracy.

 2. MOVIELENS (real ratings, optional)
    Download "ml-latest-small" from https://grouplens.org/datasets/movielens/ and run
        python evaluate.py --movielens path/to/ml-latest-small
    Users' earliest liked movies build the profile; their latest liked movies are the test set.

Metrics (all @K): precision, recall, hit-rate, NDCG, plus list diversity and catalog coverage.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

from catalog import Lookup, Movie, load_catalog
from recommender import Recommender

# --------------------------------------------------------------------------
# Hand-written taste personas (titles must exist in data/movies.csv)
# --------------------------------------------------------------------------
PERSONAS: dict[str, list[str]] = {
    "Mind-bending sci-fi fan": ["Inception", "Interstellar", "The Matrix", "Arrival", "Blade Runner 2049",
                                "Eternal Sunshine of the Spotless Mind", "The Prestige", "Memento", "Dune", "Gravity"],
    "Crime & thriller buff": ["The Godfather", "Goodfellas", "Pulp Fiction", "Se7en", "The Silence of the Lambs",
                              "The Departed", "Fight Club", "The Dark Knight", "Parasite"],
    "Pixar & family animation": ["Toy Story", "Finding Nemo", "Inside Out", "Up", "Coco", "WALL-E",
                                 "The Lion King", "Ratatouille", "Zootopia", "The Incredibles"],
    "Feel-good Bollywood": ["3 Idiots", "Taare Zameen Par", "Dangal", "Zindagi Na Milegi Dobara", "Chhichhore",
                            "Dil Chahta Hai", "Swades", "PK", "Queen", "Barfi!"],
    "Action spectacle": ["RRR", "Baahubali: The Beginning", "KGF: Chapter 1", "Vikram", "Mad Max: Fury Road",
                         "John Wick", "Mission: Impossible - Fallout", "Top Gun: Maverick", "Gladiator"],
    "Horror nights": ["Get Out", "The Conjuring", "A Quiet Place", "Hereditary", "The Shining", "Psycho",
                      "The Exorcist", "The Silence of the Lambs"],
    "Romance lover": ["The Notebook", "Pride & Prejudice", "La La Land", "Titanic", "Your Name",
                      "Crazy Rich Asians", "Barfi!", "Dil Chahta Hai"],
    "Comedy fan": ["The Hangover", "Superbad", "Bridesmaids", "Groundhog Day", "The Grand Budapest Hotel",
                   "Knives Out", "Shrek", "Crazy Rich Asians", "Guardians of the Galaxy"],
    "Epic adventure & fantasy": ["The Lord of the Rings: The Fellowship of the Ring",
                                 "Harry Potter and the Sorcerer's Stone", "Star Wars: A New Hope", "Jurassic Park",
                                 "Back to the Future", "Avatar", "Baahubali: The Beginning", "Spirited Away", "Dune"],
    "Superhero fan": ["Avengers: Endgame", "Iron Man", "Spider-Man: Into the Spider-Verse", "The Dark Knight",
                      "The Incredibles", "Black Panther", "Guardians of the Galaxy", "Avatar"],
    "True stories & inspiration": ["The Shawshank Redemption", "Forrest Gump", "A Beautiful Mind",
                                   "The Pursuit of Happyness", "The Social Network", "Moneyball", "Whiplash",
                                   "Oppenheimer", "Rocky", "Dangal"],
    "War & history": ["Saving Private Ryan", "1917", "Schindler's List", "Dunkirk", "Braveheart", "Gladiator",
                      "Oppenheimer", "Lagaan"],
}

# A "ranker" takes (liked movie ids, k) and returns a list of k movie ids.
Ranker = Callable[[list[int], int], list[int]]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def precision_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_rate_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if set(recs[:k]) & relevant else 0.0


def ndcg_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, m in enumerate(recs[:k]) if m in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def list_diversity(recs: list[int], rec: Recommender) -> float:
    """1 - average pairwise similarity. Higher = a more varied list."""
    pairs = [(a, b) for i, a in enumerate(recs) for b in recs[i + 1:]]
    return 1 - sum(rec.similarity(a, b) for a, b in pairs) / len(pairs) if pairs else 0.0


# --------------------------------------------------------------------------
# Rankers (the models being compared)
# --------------------------------------------------------------------------
def make_rankers(rec: Recommender, seed: int = 0) -> dict[str, Ranker]:
    rng = random.Random(seed)
    ids = [m.id for m in rec.movies]
    genres = {m.id: set(m.genres) for m in rec.movies}

    def random_ranker(liked, k):
        pool = [i for i in ids if i not in liked]
        return rng.sample(pool, min(k, len(pool)))

    def genre_ranker(liked, k):                    # baseline: average genre overlap only
        scored = []
        for i in ids:
            if i in liked:
                continue
            overlap = sum(len(genres[i] & genres[l]) / len(genres[i] | genres[l]) for l in liked) / len(liked)
            scored.append((-overlap, i))
        return [i for _, i in sorted(scored)[:k]]

    def content(liked, k):
        return [r.movie.id for r in rec.recommend({m: 5 for m in liked}, n=k)]

    def content_diverse(liked, k):
        return [r.movie.id for r in rec.recommend({m: 5 for m in liked}, n=k, diverse=True)]

    return {"Random": random_ranker, "Genre overlap only": genre_ranker,
            "Content-based (TF-IDF)": content, "Content-based + diversity": content_diverse}


def _summarise(rows: list[dict], coverage_ids: set[int], catalog_size: int) -> dict[str, float]:
    keys = rows[0].keys()
    out = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
    out["coverage"] = round(len(coverage_ids) / catalog_size, 4)
    return out


# --------------------------------------------------------------------------
# 1) Persona benchmark
# --------------------------------------------------------------------------
def persona_benchmark(movies: Optional[list[Movie]] = None, k: int = 10, given: int = 3,
                      repeats: int = 20, seed: int = 7) -> dict[str, dict[str, float]]:
    movies = movies or load_catalog()
    lookup, rec = Lookup(movies), Recommender(movies)
    persona_ids = {}
    for name, titles in PERSONAS.items():
        found = [lookup.find(t)[0] for t in titles]
        for want, got in zip(titles, found):
            if got.title != want:
                raise ValueError(f"Persona '{name}': '{want}' is not in the catalog")
        persona_ids[name] = [m.id for m in found]

    results = {}
    for model, ranker in make_rankers(rec, seed).items():
        rng = random.Random(seed)
        rows, covered = [], set()
        for name, ids in persona_ids.items():
            for _ in range(repeats):
                liked = rng.sample(ids, given)                  # the few movies the bot "knows" about
                hidden = set(ids) - set(liked)                  # everything else they love
                recs = ranker(liked, k)
                covered.update(recs)
                rows.append({f"precision@{k}": precision_at_k(recs, hidden, k),
                             f"recall@{k}": recall_at_k(recs, hidden, k),
                             f"hit_rate@{k}": hit_rate_at_k(recs, hidden, k),
                             f"ndcg@{k}": ndcg_at_k(recs, hidden, k),
                             "diversity": list_diversity(recs, rec)})
        results[model] = _summarise(rows, covered, len(movies))
    return results


# --------------------------------------------------------------------------
# 2) MovieLens (real ratings)
# --------------------------------------------------------------------------
def load_movielens(folder: Path | str):
    folder = Path(folder)
    movies = []
    with open(folder / "movies.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = re.search(r"\((\d{4})\)\s*$", row["title"])
            genres = () if row["genres"].startswith("(no") else tuple(row["genres"].split("|"))
            movies.append(Movie(int(row["movieId"]), re.sub(r"\s*\(\d{4}\)\s*$", "", row["title"]),
                                int(m.group(1)) if m else 0, genres))
    ratings: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    with open(folder / "ratings.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ratings[int(row["userId"])].append((int(row["timestamp"]), int(row["movieId"]), float(row["rating"])))
    return movies, ratings


def movielens_benchmark(folder: Path | str, k: int = 10, max_users: int = 200, min_likes: int = 10,
                        seed: int = 7) -> dict[str, dict[str, float]]:
    movies, ratings = load_movielens(folder)
    rec = Recommender(movies)
    valid = {m.id for m in movies}
    popularity = Counter(mid for rs in ratings.values() for _, mid, r in rs if r >= 4 and mid in valid)
    rng = random.Random(seed)
    users = [u for u, rs in ratings.items() if sum(r >= 4 for _, _, r in rs) >= min_likes]
    rng.shuffle(users)
    users = users[:max_users]
    if not users:
        raise ValueError("No users with enough liked movies - lower min_likes.")

    ids = [m.id for m in movies]
    models = {
        "Random": lambda train, k: rng.sample([i for i in ids if i not in train], k),
        "Popularity": lambda train, k: [i for i, _ in popularity.most_common(k + len(train)) if i not in train][:k],
        "Content-based (TF-IDF)": lambda train, k: [r.movie.id for r in rec.recommend(train, n=k)],
    }
    results = {}
    for model, ranker in models.items():
        rows, covered = [], set()
        for u in users:
            history = sorted(r for r in ratings[u] if r[1] in valid)          # oldest first
            likes = [mid for _, mid, r in history if r >= 4]
            cut = int(len(likes) * 0.8)
            test = set(likes[cut:])
            first_test_time = next(t for t, mid, r in history if mid == likes[cut])
            train = {mid: r for t, mid, r in history if t < first_test_time}
            if not test or not any(r >= 4 for r in train.values()):
                continue
            recs = ranker(train, k)
            covered.update(recs)
            rows.append({f"precision@{k}": precision_at_k(recs, test, k), f"recall@{k}": recall_at_k(recs, test, k),
                         f"hit_rate@{k}": hit_rate_at_k(recs, test, k), f"ndcg@{k}": ndcg_at_k(recs, test, k),
                         "diversity": list_diversity(recs, rec)})
        if rows:
            results[model] = _summarise(rows, covered, len(movies))
    return results


# --------------------------------------------------------------------------
# Pretty printing / CLI
# --------------------------------------------------------------------------
def format_table(results: dict[str, dict[str, float]]) -> str:
    cols = list(next(iter(results.values())).keys())
    width = max(len(m) for m in results) + 2
    lines = ["Model".ljust(width) + "".join(c.rjust(14) for c in cols)]
    lines.append("-" * len(lines[0]))
    for model, vals in results.items():
        lines.append(model.ljust(width) + "".join(f"{vals[c]:14.3f}" for c in cols))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation quality")
    parser.add_argument("--k", type=int, default=10, help="list length (default 10)")
    parser.add_argument("--movielens", help="folder with MovieLens movies.csv and ratings.csv")
    parser.add_argument("--max-users", type=int, default=200)
    args = parser.parse_args()

    if args.movielens:
        print(f"MovieLens benchmark (top-{args.k}, up to {args.max_users} users)\n")
        print(format_table(movielens_benchmark(args.movielens, args.k, args.max_users)))
    else:
        print(f"Persona benchmark: 12 personas x 20 random draws, 3 known favourites each, top-{args.k}\n")
        print(format_table(persona_benchmark(k=args.k)))
        print("\nHigher is better for every column. Random is the floor; the personas are hand-written, so "
              "use this to compare models, and run --movielens for real-world numbers.")


if __name__ == "__main__":
    main()
