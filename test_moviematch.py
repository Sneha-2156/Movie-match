"""Run with:  python -m unittest -v test_moviematch   (no internet needed)"""
import csv
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import evaluate
from catalog import Lookup, load_catalog
from recommender import RecommendationError, Recommender, dot, norm
from web_app import App, make_server

MOVIES = load_catalog()
LOOKUP = Lookup(MOVIES)
REC = Recommender(MOVIES)


def mid(title):
    return LOOKUP.find(title)[0].id


def titles(recs):
    return [r.movie.title for r in recs]


class CatalogTests(unittest.TestCase):
    def test_catalog_is_clean(self):
        self.assertGreaterEqual(len(MOVIES), 100)
        self.assertEqual(len({m.id for m in MOVIES}), len(MOVIES))
        self.assertEqual(len({m.title for m in MOVIES}), len(MOVIES))
        self.assertTrue(all(m.genres and m.year > 1900 for m in MOVIES))

    def test_fuzzy_title_search(self):
        self.assertEqual(LOOKUP.find("inceptoin")[0].title, "Inception")
        self.assertEqual(LOOKUP.find("3 idiots")[0].title, "3 Idiots")
        self.assertEqual(LOOKUP.find("lord of the rings")[0].title,
                         "The Lord of the Rings: The Fellowship of the Ring")
        self.assertEqual(LOOKUP.find("zzzzqq"), [])

    def test_genre_and_language_lookup(self):
        self.assertEqual(LOOKUP.genre("scifi"), "Sci-Fi")
        self.assertEqual(LOOKUP.genre("science fiction"), "Sci-Fi")
        self.assertEqual(LOOKUP.genre("horrer"), "Horror")
        self.assertIsNone(LOOKUP.genre("nonsense-genre"))
        self.assertEqual(LOOKUP.language("kannda"), "Kannada")


class RecommenderTests(unittest.TestCase):
    def test_vectors_are_unit_length(self):
        for vec in REC.vectors.values():
            self.assertAlmostEqual(norm(vec), 1.0, places=6)

    def test_similar_to_inception(self):
        top = titles(REC.similar(mid("Inception"), n=6))
        self.assertIn("The Matrix", top[:3])
        self.assertTrue({"The Prestige", "Interstellar", "Memento"} & set(top))
        self.assertNotIn("Inception", top)

    def test_cross_language_taste(self):
        top = titles(REC.similar(mid("3 Idiots"), n=5))
        self.assertTrue({"Chhichhore", "Taare Zameen Par", "PK"} <= set(top))

    def test_never_recommends_rated_movies(self):
        ratings = {mid("Inception"): 5, mid("Interstellar"): 5, mid("Dune"): 1}
        top = titles(REC.recommend(ratings, n=20))
        for t in ("Inception", "Interstellar", "Dune"):
            self.assertNotIn(t, top)

    def test_dislikes_push_results_away(self):
        liked_only = titles(REC.recommend({mid("Inception"): 5}, n=10))
        with_dislike = titles(REC.recommend({mid("Inception"): 5, mid("The Matrix"): 1}, n=10))
        self.assertIn("The Matrix", liked_only)
        self.assertNotIn("The Matrix", with_dislike)

    def test_genre_only_cold_start(self):
        recs = REC.recommend(genres=["Horror"], n=5)
        self.assertTrue(all("Horror" in r.movie.genres for r in recs))

    def test_filters(self):
        base = {mid("Inception"): 5}
        self.assertTrue(all(r.movie.year >= 2015 for r in REC.recommend(base, n=8, year_min=2015)))
        self.assertTrue(all(r.movie.language == "Hindi" for r in REC.recommend(base, n=8, languages=["Hindi"])))
        self.assertTrue(all("Thriller" not in r.movie.genres for r in REC.recommend(base, n=15, exclude_genres=["Thriller"])))
        self.assertEqual(REC.recommend(base, n=5, year_min=2100), [])

    def test_explanations(self):
        rec = REC.similar(mid("Toy Story"), n=1)[0]
        self.assertEqual(rec.because, "Toy Story")
        self.assertIn("Because you liked Toy Story", rec.reasons())
        self.assertTrue(rec.shared)

    def test_diversity_option_adds_variety(self):
        ratings = {mid("The Dark Knight"): 5, mid("Inception"): 5}
        plain = [r.movie.id for r in REC.recommend(ratings, n=8)]
        mixed = [r.movie.id for r in REC.recommend(ratings, n=8, diverse=True)]
        self.assertGreater(evaluate.list_diversity(mixed, REC), evaluate.list_diversity(plain, REC))

    def test_errors(self):
        with self.assertRaises(RecommendationError):
            REC.recommend({}, [])
        with self.assertRaises(RecommendationError):
            REC.recommend({99999: 5})
        with self.assertRaises(RecommendationError):
            REC.recommend({mid("Inception"): 1})            # only a dislike: nothing positive to go on


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        recs, relevant = [1, 2, 3, 4], {2, 4, 9}
        self.assertAlmostEqual(evaluate.precision_at_k(recs, relevant, 4), 0.5)
        self.assertAlmostEqual(evaluate.recall_at_k(recs, relevant, 4), 2 / 3)
        self.assertEqual(evaluate.hit_rate_at_k(recs, relevant, 1), 0.0)
        self.assertEqual(evaluate.hit_rate_at_k(recs, relevant, 2), 1.0)
        self.assertAlmostEqual(evaluate.ndcg_at_k([2, 4, 9], {2, 4, 9}, 3), 1.0)
        self.assertLess(evaluate.ndcg_at_k([1, 3, 2], {2}, 3), 1.0)

    def test_persona_titles_exist_and_benchmark_beats_random(self):
        results = evaluate.persona_benchmark(MOVIES, repeats=5)
        content, rand = results["Content-based (TF-IDF)"], results["Random"]
        self.assertGreater(content["precision@10"], rand["precision@10"] * 3)
        self.assertGreater(content["ndcg@10"], rand["ndcg@10"] * 3)
        self.assertGreater(results["Content-based + diversity"]["diversity"], content["diversity"])


def write_fake_movielens(folder: Path, users=30, per_group=15):
    """Tiny MovieLens-shaped dataset: 4 genre groups, each user loves one group."""
    groups = ["Action", "Comedy", "Horror", "Romance"]
    with open(folder / "movies.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["movieId", "title", "genres"])
        for g, name in enumerate(groups):
            for i in range(per_group):
                w.writerow([g * per_group + i + 1, f"{name} Movie {i} ({1990 + i})", f"{name}|Drama" if i % 3 == 0 else name])
    with open(folder / "ratings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["userId", "movieId", "rating", "timestamp"])
        for u in range(users):
            fav, t = u % 4, 0
            for i in range(per_group * 3):                 # 45 ratings, mostly of the favourite group
                group = fav if i % 3 else (fav + 1 + i // 3 % 3) % 4
                movie = group * per_group + (i * 7 + u) % per_group + 1
                w.writerow([u + 1, movie, 5 if group == fav else 1, 1000 + t])
                t += 1


class MovieLensTests(unittest.TestCase):
    def test_loader_and_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fake_movielens(Path(tmp))
            movies, ratings = evaluate.load_movielens(tmp)
            self.assertEqual(len(movies), 60)
            self.assertEqual(movies[0].year, 1990)
            self.assertEqual(movies[0].title, "Action Movie 0")
            results = evaluate.movielens_benchmark(tmp, k=5, max_users=20, min_likes=10)
            self.assertGreater(results["Content-based (TF-IDF)"]["precision@5"],
                               results["Random"]["precision@5"] * 2)


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(App(MOVIES), port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, payload, headers=None, raw=None):
        req = urllib.request.Request(self.base + "/api/recommend", data=raw or json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_page_catalog_and_evaluation(self):
        with urllib.request.urlopen(self.base + "/") as resp:
            self.assertIn(b"MovieMatch", resp.read())
        with urllib.request.urlopen(self.base + "/api/catalog") as resp:
            data = json.loads(resp.read())
        self.assertEqual(len(data["movies"]), len(MOVIES))
        self.assertIn("Sci-Fi", data["genres"])
        with urllib.request.urlopen(self.base + "/api/evaluate") as resp:
            self.assertIn("Content-based (TF-IDF)", json.loads(resp.read()))

    def test_recommend_endpoint(self):
        status, body = self.post({"ratings": {str(mid("Inception")): 5}, "n": 5, "year_min": 2000})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["results"]), 5)
        self.assertIn("Because you liked Inception", body["results"][0]["reason"])
        status, body = self.post({"genres": ["scifi"], "languages": ["Hindi"], "n": 3})
        self.assertEqual(status, 200)

    def test_bad_requests(self):
        self.assertEqual(self.post({})[0], 400)                                   # no taste information
        self.assertEqual(self.post({"ratings": {"999999": 5}})[0], 400)
        self.assertEqual(self.post({"ratings": {"abc": 5}})[0], 400)
        self.assertEqual(self.post(None, raw=b"not json")[0], 400)
        self.assertEqual(self.post({}, headers={"Content-Type": "text/plain"})[0], 415)

    def test_foreign_host_rejected(self):
        req = urllib.request.Request(self.base + "/api/catalog", headers={"Host": "evil.example.com"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
