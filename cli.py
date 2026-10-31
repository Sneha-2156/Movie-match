"""
cli.py - MovieMatch in the terminal.

  python cli.py                                       interactive session
  python cli.py --genres sci-fi thriller -n 8          recommend from favourite genres
  python cli.py --liked "Inception" "3 Idiots"         recommend from movies you liked
  python cli.py --similar "Toy Story" --diverse        "more like this"
  python cli.py --evaluate                             measure recommendation quality
"""
from __future__ import annotations

import argparse
import os
import sys

from catalog import Lookup, Movie, load_catalog
from evaluate import format_table, persona_benchmark
from recommender import RecommendationError, Recommender

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
CYAN, GREEN, YELLOW, RED = "\033[96m", "\033[92m", "\033[93m", "\033[91m"

HELP = f"""{BOLD}Build your taste profile, then ask for recommendations{RESET}
  genres sci-fi, thriller      add favourite genres          (genres clear to reset)
  like <title>                 movie you enjoyed   (5 stars)
  dislike <title>              movie you didn't    (1 star)
  rate <title> = 1-5           any rating
  search <text>                find titles in the catalog
  year 2000-2020               release-year filter  (year off)
  language hindi               language filter      (language off)
  exclude horror               never show this genre (exclude off)
  diverse on|off               mix things up a little
  similar <title>              movies like one title
  recommend [n]   or  go       show recommendations (default 8)
  profile  /  reset  /  evaluate  /  help  /  exit"""


def setup_terminal():
    os.system("")                                       # ANSI colours on Windows 10+
    for stream in (sys.stdout, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def show(recs, header: str = ""):
    if header:
        print(f"\n{BOLD}{header}{RESET}")
    for i, r in enumerate(recs, 1):
        m = r.movie
        bar = "#" * max(1, round(r.score * 30))
        print(f"\n {CYAN}{BOLD}{i:>2}. {m.title}{RESET} ({m.year})  {DIM}{', '.join(m.genres)} | {m.language}{RESET}")
        print(f"     {GREEN}{bar}{RESET} {r.score:.2f}   {DIM}{r.reasons()}{RESET}")
    print()


class Session:
    def __init__(self):
        self.movies = load_catalog()
        self.lookup = Lookup(self.movies)
        self.rec = Recommender(self.movies)
        self.ratings: dict[int, float] = {}
        self.genres: list[str] = []
        self.filters: dict = {"year_min": None, "year_max": None, "languages": [], "exclude_genres": []}
        self.diverse = False

    # --------------------------------------------------------------- helpers
    def pick_movie(self, query: str) -> Movie | None:
        matches = self.lookup.find(query)
        if not matches:
            print(f"{YELLOW}No movie found for '{query}'. Try: search {query.split()[0] if query else 'matrix'}{RESET}")
            return None
        best = matches[0]
        if len(matches) > 1 and best.title.lower() != query.strip().lower():
            others = ", ".join(m.title for m in matches[1:3])
            print(f"{DIM}Using '{best.title}' (other matches: {others}){RESET}")
        return best

    def recommend(self, n: int = 8):
        try:
            recs = self.rec.recommend(self.ratings, self.genres, n=n, diverse=self.diverse, **self.filters)
        except RecommendationError as exc:
            print(f"{YELLOW}{exc}{RESET}")
            return
        if not recs:
            print(f"{YELLOW}Nothing matches those filters - try loosening them.{RESET}")
            return
        show(recs, "Recommended for you")

    def profile(self):
        liked = [self.rec.by_id[i].title for i, r in self.ratings.items() if r >= 4]
        disliked = [self.rec.by_id[i].title for i, r in self.ratings.items() if r <= 2]
        print(f"  Genres   : {', '.join(self.genres) or '-'}")
        print(f"  Liked    : {', '.join(liked) or '-'}")
        print(f"  Disliked : {', '.join(disliked) or '-'}")
        f = self.filters
        year = f"{f['year_min'] or '...'}-{f['year_max'] or '...'}" if f["year_min"] or f["year_max"] else "any"
        print(f"  Filters  : year {year}, language {', '.join(f['languages']) or 'any'}, "
              f"exclude {', '.join(f['exclude_genres']) or 'none'}, diverse {'on' if self.diverse else 'off'}")

    # --------------------------------------------------------------- commands
    def handle(self, line: str) -> bool:
        cmd, _, arg = line.strip().partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        if cmd in ("exit", "quit", "q"):
            return False
        if cmd == "help":
            print(HELP)
        elif cmd == "genres":
            if arg.lower() == "clear":
                self.genres = []
            parts = [p.strip() for p in arg.split(",") if p.strip()]
            if len(parts) == 1 and not self.lookup.genre(parts[0]):
                parts = parts[0].split()                       # "sci-fi thriller" -> two genres
            for part in parts:
                g = self.lookup.genre(part)
                if g and g not in self.genres:
                    self.genres.append(g)
                elif not g and part.lower() != "clear":
                    print(f"{YELLOW}Unknown genre '{part}'. Available: {', '.join(self.lookup.genres)}{RESET}")
            print(f"{DIM}Genres: {', '.join(self.genres) or '-'}{RESET}")
        elif cmd in ("like", "dislike", "rate"):
            query, stars = arg, 5 if cmd == "like" else 1
            if cmd == "rate":
                query, _, num = arg.rpartition("=")
                if not num.strip().isdigit() or not 1 <= int(num) <= 5:
                    print(f"{YELLOW}Use: rate <title> = 1-5{RESET}")
                    return True
                stars = int(num)
            movie = self.pick_movie(query)
            if movie:
                self.ratings[movie.id] = stars
                print(f"{GREEN}Rated '{movie.title}' {stars}/5.{RESET}")
        elif cmd == "search":
            for m in self.lookup.find(arg, limit=8):
                print(f"  {m.title} ({m.year}) {DIM}{', '.join(m.genres)}{RESET}")
        elif cmd == "year":
            if arg.lower() == "off" or not arg:
                self.filters["year_min"] = self.filters["year_max"] = None
            else:
                lo, _, hi = arg.partition("-")
                try:
                    self.filters["year_min"] = int(lo) if lo.strip() else None
                    self.filters["year_max"] = int(hi) if hi.strip() else None
                except ValueError:
                    print(f"{YELLOW}Use: year 2000-2020{RESET}")
        elif cmd == "language":
            if arg.lower() in ("off", ""):
                self.filters["languages"] = []
            else:
                lang = self.lookup.language(arg)
                self.filters["languages"] = [lang] if lang else []
                if not lang:
                    print(f"{YELLOW}Unknown language. Available: {', '.join(self.lookup.languages)}{RESET}")
        elif cmd == "exclude":
            if arg.lower() in ("off", ""):
                self.filters["exclude_genres"] = []
            else:
                g = self.lookup.genre(arg)
                if g:
                    self.filters["exclude_genres"].append(g)
                else:
                    print(f"{YELLOW}Unknown genre '{arg}'.{RESET}")
        elif cmd == "diverse":
            self.diverse = arg.lower() in ("on", "yes", "1", "true")
            print(f"{DIM}Diversity is {'on' if self.diverse else 'off'}.{RESET}")
        elif cmd == "similar":
            movie = self.pick_movie(arg)
            if movie:
                show(self.rec.similar(movie.id, n=8, diverse=self.diverse, **self.filters), f"More like '{movie.title}'")
        elif cmd in ("recommend", "go", "rec", ""):
            self.recommend(int(arg) if arg.isdigit() else 8)
        elif cmd == "profile":
            self.profile()
        elif cmd == "reset":
            self.__init__()
            print(f"{DIM}Profile cleared.{RESET}")
        elif cmd == "evaluate":
            print(format_table(persona_benchmark(self.movies)))
        else:
            print(f"{YELLOW}Unknown command. Type 'help'.{RESET}")
        return True


def interactive():
    print(f"{CYAN}{BOLD}MovieMatch{RESET}  {DIM}SAM AI Technologies | AI Internship | Task 3{RESET}")
    print(f"{DIM}Tell me what you like, then type 'go'. Type 'help' for commands.{RESET}\n")
    session = Session()
    print(f"{DIM}{len(session.movies)} movies loaded. Example: genres sci-fi, thriller   |   like Inception   |   go{RESET}\n")
    while True:
        try:
            line = input(f"{CYAN}>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not session.handle(line):
            return


def main():
    parser = argparse.ArgumentParser(description="MovieMatch - content-based movie recommendations")
    parser.add_argument("--genres", nargs="*", default=[], help="favourite genres")
    parser.add_argument("--liked", nargs="*", default=[], help="titles you liked")
    parser.add_argument("--disliked", nargs="*", default=[], help="titles you disliked")
    parser.add_argument("--similar", help="find movies similar to this title")
    parser.add_argument("-n", type=int, default=8, help="how many results (default 8)")
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--language", help="only this language")
    parser.add_argument("--exclude", nargs="*", default=[], help="genres to avoid")
    parser.add_argument("--diverse", action="store_true", help="add variety to the list")
    parser.add_argument("--evaluate", action="store_true", help="run the quality benchmark")
    args = parser.parse_args()

    setup_terminal()
    if args.evaluate:
        print(format_table(persona_benchmark()))
        return 0
    if not (args.genres or args.liked or args.similar):
        interactive()
        return 0

    s = Session()
    for query, stars in [(t, 5) for t in args.liked] + [(t, 1) for t in args.disliked]:
        movie = s.pick_movie(query)
        if movie:
            s.ratings[movie.id] = stars
    s.genres = [g for g in (s.lookup.genre(x) for x in args.genres) if g]
    s.diverse = args.diverse
    s.filters.update(year_min=args.year_min, year_max=args.year_max,
                     languages=[s.lookup.language(args.language)] if args.language and s.lookup.language(args.language) else [],
                     exclude_genres=[g for g in (s.lookup.genre(x) for x in args.exclude) if g])
    if args.similar:
        movie = s.pick_movie(args.similar)
        if movie:
            show(s.rec.similar(movie.id, n=args.n, diverse=s.diverse, **s.filters), f"More like '{movie.title}'")
    else:
        s.recommend(args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
