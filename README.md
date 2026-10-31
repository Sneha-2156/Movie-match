# MovieMatch - Recommendation System

**SAM AI Technologies - AI Internship - Task 3: Recommendation System**

A content-based movie recommender written in pure Python (no numpy, no scikit-learn): TF-IDF vectors, cosine similarity, user profiles with likes and dislikes, filters, explanations and a built-in evaluation. Use it from the **browser** or the **terminal**.

## Task requirements -> how MovieMatch meets them

| Requirement | Implementation |
|---|---|
| Build a system to suggest items (movies/products) | 105-movie catalog (`data/movies.csv`): Hollywood, Bollywood, Telugu, Kannada, Tamil, anime, Korean |
| Use user preferences as input | Favourite genres, liked / disliked movies, year, language and genre filters |
| Simple filtering (content-based) | TF-IDF over genres, themes, director, language, decade + cosine similarity |
| Show recommended items | Ranked list with similarity score and a plain-English reason for each pick |
| Evaluate recommendation quality | Precision, recall, hit-rate, NDCG, diversity, coverage vs. baselines (`evaluate.py`) |

## Features

- **Taste profile**: likes add to your profile, dislikes subtract from it, typed-in genres count too.
- **Explainable**: "Because you liked Inception - shares Sci-Fi, mind bending, director Christopher Nolan".
- **Filters**: release year, language, genres to avoid.
- **Variety control**: optional MMR re-ranking ("Mix it up") trades a little relevance for a more varied list.
- **More like this**: one click to find movies similar to any result.
- **Forgiving search**: `inceptoin`, `scifi`, `kannda` still work.
- **Evaluation tab** in the web app, plus real-data evaluation on MovieLens.
- **20 unit tests**, local-only server (127.0.0.1, JSON only, Host-header check).

## Run it

```bash
python web_app.py                                     # browser app at http://127.0.0.1:8000
python cli.py                                         # interactive terminal session
python cli.py --genres sci-fi thriller -n 8           # one-shot from genres
python cli.py --liked "Inception" "3 Idiots" --year-min 2000
python cli.py --similar "Toy Story" --diverse         # "more like this"
python evaluate.py                                    # quality benchmark
python -m unittest -v test_moviematch                 # tests
```

Terminal session example:

```
> genres comedy, drama
> like 3 idiots
> language hindi
> go
```

## How it works

```
movie -> tokens (genre, theme, director, language, decade) -> weight = field weight x IDF -> unit vector
user  -> profile = sum( (rating-3)/2 x movie vector ) + genre vector
score = cosine(profile, movie)  ->  filter  ->  rank (optionally MMR)  ->  explain
```

- **IDF** makes rare traits count more (a shared director says more than a shared "English" tag).
- **Field weights** (`FIELD_WEIGHTS` in `recommender.py`): genre 3, theme 2, director 1.5, language 1, decade 0.5.
- **Cold start**: genres alone are enough to get started.

## Evaluation

`python evaluate.py` runs a persona benchmark: 12 hand-written taste personas (sci-fi, crime, Pixar, Bollywood, horror, ...). The recommender is told only **3** of a persona's favourites and must find the rest in its top 10 (20 random draws per persona).

| Model | precision@10 | recall@10 | hit-rate@10 | NDCG@10 | diversity | coverage |
|---|---|---|---|---|---|---|
| Random | 0.059 | 0.098 | 0.454 | 0.079 | 0.940 | 1.000 |
| Genre overlap only | 0.393 | 0.667 | 0.996 | 0.640 | 0.827 | 0.971 |
| **Content-based (TF-IDF)** | **0.403** | **0.684** | 0.992 | **0.654** | 0.823 | 0.962 |
| Content-based + diversity | 0.390 | 0.662 | 0.996 | 0.633 | 0.854 | 0.991 |

How to read it:
- Content-based is ~7x better than random at precision@10 and finds ~68% of a persona's hidden favourites.
- It is only slightly better than genre-only matching here, because the personas are largely genre-driven.
- The diversity option costs ~1 point of precision for noticeably more varied lists - a classic trade-off.
- **Caveat:** personas are hand-written, so this compares models against each other; it does not predict real-world accuracy.

### Real data: MovieLens

1. Download `ml-latest-small` from https://grouplens.org/datasets/movielens/
2. Run `python evaluate.py --movielens path/to/ml-latest-small`

Each user's earliest 80% of liked movies build the profile; the latest 20% are the test set. Compared against Random and Popularity baselines. (MovieLens only has genres, so expect lower numbers than the persona benchmark.)

## Project structure

```
moviematch/
  data/movies.csv      # the catalog - add rows to grow it
  catalog.py           # Movie model, CSV loader, fuzzy title/genre lookup
  recommender.py       # TF-IDF vectors, profiles, filters, MMR, explanations
  evaluate.py          # metrics, personas benchmark, MovieLens benchmark
  cli.py               # terminal app
  web_app.py           # local web server (stdlib)
  static/index.html    # browser UI
  test_moviematch.py
```

## Ideas to extend

- Add collaborative filtering (user-user or item-item) and blend it with content scores
- Use plot summaries with TF-IDF over words
- Swap the catalog for products (same pipeline: categories, brand, tags)

---
Built by Sneha for the SAM AI Technologies internship program.
