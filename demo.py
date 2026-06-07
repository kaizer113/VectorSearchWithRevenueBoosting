#!/usr/bin/env python3
"""
Redis Vector Search revenue-boosting demo using Redis-py.

Creates:
  - 5000 movie vectors in the movies: namespace
  - 10 user vectors in the users: namespace

Then recommends the top 3 movies per user with one Redis command:
  FT.AGGREGATE idx:movies ... KNN ... APPLY boosted_score ... SORTBY ...
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import struct
import time
from dataclasses import dataclass
from typing import Iterable

import redis
from redis.exceptions import ResponseError


DIMENSIONS = 16
MOVIE_COUNT = 5000
USER_COUNT = 10
MOVIE_INDEX = "idx:movies"
LEGACY_MOVIE_HNSW_INDEX = "idx:movies_hnsw"
USER_INDEX = "idx:users"
MOVIE_PREFIX = "movies:"
USER_PREFIX = "users:"

AXES = [
    "action",
    "comedy",
    "romance",
    "sci-fi",
    "fantasy",
    "drama",
    "horror",
    "crime",
    "animation",
    "documentary",
    "family",
    "thriller",
    "adventure",
    "music",
    "history",
    "sports",
]

GENRE_PROFILES = {
    "Action": {"action": 1.0, "thriller": 0.45, "adventure": 0.35},
    "Adventure": {"adventure": 1.0, "action": 0.4, "fantasy": 0.25},
    "Animation": {"animation": 1.0, "family": 0.55, "comedy": 0.25},
    "Comedy": {"comedy": 1.0, "romance": 0.22, "family": 0.18},
    "Crime": {"crime": 1.0, "thriller": 0.55, "drama": 0.25},
    "Documentary": {"documentary": 1.0, "history": 0.35, "sports": 0.2},
    "Drama": {"drama": 1.0, "romance": 0.25, "history": 0.15},
    "Family": {"family": 1.0, "animation": 0.45, "adventure": 0.2},
    "Fantasy": {"fantasy": 1.0, "adventure": 0.45, "romance": 0.15},
    "Historical": {"history": 1.0, "drama": 0.45, "documentary": 0.2},
    "Horror": {"horror": 1.0, "thriller": 0.45, "crime": 0.15},
    "Music": {"music": 1.0, "drama": 0.25, "romance": 0.15},
    "Romance": {"romance": 1.0, "comedy": 0.25, "drama": 0.25},
    "Sci-Fi": {"sci-fi": 1.0, "action": 0.38, "thriller": 0.25},
    "Sports": {"sports": 1.0, "drama": 0.35, "documentary": 0.2},
    "Thriller": {"thriller": 1.0, "crime": 0.35, "action": 0.3},
}

GENRE_PAIRINGS = [
    ("Action", "Sci-Fi"),
    ("Action", "Thriller"),
    ("Adventure", "Fantasy"),
    ("Animation", "Family"),
    ("Comedy", "Romance"),
    ("Crime", "Drama"),
    ("Crime", "Thriller"),
    ("Documentary", "Historical"),
    ("Drama", "Historical"),
    ("Drama", "Romance"),
    ("Fantasy", "Romance"),
    ("Horror", "Thriller"),
    ("Music", "Drama"),
    ("Sci-Fi", "Thriller"),
    ("Sports", "Drama"),
    ("Adventure", "Comedy"),
]

MOVIE_NOUNS = [
    "Signal",
    "Harbor",
    "Orbit",
    "Promise",
    "Shadow",
    "Circuit",
    "Kingdom",
    "Finale",
    "Journey",
    "Legacy",
    "Velocity",
    "Memory",
    "Mirage",
    "Frontier",
    "Echo",
    "Rival",
    "Garden",
    "Equation",
    "Festival",
    "Witness",
]

MOVIE_ADJECTIVES = [
    "Neon",
    "Hidden",
    "Silver",
    "Brave",
    "Last",
    "Quiet",
    "Midnight",
    "Golden",
    "Electric",
    "Wild",
    "Crimson",
    "Northern",
    "Impossible",
    "Second",
    "Paper",
    "Burning",
    "Infinite",
    "Velvet",
    "Borrowed",
    "Famous",
]

DESCRIPTION_TEMPLATES = [
    "A {tone} story about {subject} where {conflict}.",
    "When {event}, {subject} must choose between ambition and belonging.",
    "{subject} discovers that {secret}, turning a familiar world inside out.",
    "A {tone} ensemble follows {subject} through a season of risk, loyalty, and reinvention.",
]

TONES = [
    "high-energy",
    "warm",
    "tense",
    "witty",
    "sweeping",
    "intimate",
    "stylish",
    "thoughtful",
]

SUBJECTS = [
    "a retired pilot",
    "two rival chefs",
    "a teenage coder",
    "an unlikely team",
    "a small-town coach",
    "a museum archivist",
    "a stranded family",
    "a reluctant detective",
    "an ambitious singer",
    "a crew of old friends",
]

CONFLICTS = [
    "a buried secret threatens everything they built",
    "the clock runs out before the truth can surface",
    "success costs more than anyone expected",
    "the safest choice becomes the most dangerous one",
    "a public failure becomes a private turning point",
]

EVENTS = [
    "a citywide blackout exposes a conspiracy",
    "an old recording resurfaces",
    "a championship collapses in the final minute",
    "a prototype starts predicting impossible events",
    "a forgotten festival returns after fifty years",
]

SECRETS = [
    "the legend was engineered",
    "their rival has been protecting them",
    "the map points home instead of away",
    "the monster is a symptom, not the cause",
    "the missing evidence was hidden in plain sight",
]

USER_PERSONAS = [
    ("users:001", "Maya", ["Sci-Fi", "Action", "Thriller"], "loves smart sci-fi, fast pacing, and big cinematic stakes"),
    ("users:002", "Noah", ["Comedy", "Romance", "Music"], "likes charming, funny stories with heart and a great soundtrack"),
    ("users:003", "Ava", ["Documentary", "Historical", "Drama"], "prefers grounded true stories, history, and thoughtful character arcs"),
    ("users:004", "Liam", ["Animation", "Family", "Adventure"], "watches colorful family adventures and playful animated worlds"),
    ("users:005", "Sofia", ["Crime", "Thriller", "Drama"], "enjoys mysteries, tense investigations, and morally complex drama"),
    ("users:006", "Ethan", ["Fantasy", "Adventure", "Action"], "wants world-building, quests, magic, and heroic spectacle"),
    ("users:007", "Isabella", ["Horror", "Thriller", "Crime"], "likes suspense, dark twists, and unnerving late-night stories"),
    ("users:008", "Lucas", ["Sports", "Drama", "Documentary"], "follows comeback stories, team dynamics, and sports documentaries"),
    ("users:009", "Amelia", ["Drama", "Romance", "Historical"], "likes emotional dramas, period settings, and slow-burn relationships"),
    ("users:010", "Ben", ["Comedy", "Adventure", "Action"], "watches crowd-pleasing adventures with jokes and momentum"),
]


@dataclass(frozen=True)
class Movie:
    key: str
    movie_id: str
    title: str
    genre: str
    description: str
    revenue: float
    score: float
    embedding: list[float]


@dataclass(frozen=True)
class User:
    key: str
    user_id: str
    name: str
    watch_history: list[str]
    preferences: str
    embedding: list[float]


def stable_rng(seed: str) -> random.Random:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def vector_from_genres(genres: Iterable[str], seed: str, noise: float = 0.045) -> list[float]:
    values = [0.0] * DIMENSIONS
    for genre in genres:
        for axis, weight in GENRE_PROFILES[genre].items():
            values[AXES.index(axis)] += weight

    rng = stable_rng(seed)
    for index in range(DIMENSIONS):
        values[index] += rng.uniform(-noise, noise)

    return normalize(values)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(vector: bytes) -> list[float]:
    return list(struct.unpack(f"<{DIMENSIONS}f", vector))


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def create_movies() -> list[Movie]:
    movies = []
    for index in range(1, MOVIE_COUNT + 1):
        primary, secondary = GENRE_PAIRINGS[(index - 1) % len(GENRE_PAIRINGS)]
        if index % 7 == 0:
            primary, secondary = secondary, primary

        rng = stable_rng(f"movie:{index}")
        title = f"{MOVIE_ADJECTIVES[(index - 1) % len(MOVIE_ADJECTIVES)]} {MOVIE_NOUNS[(index * 7) % len(MOVIE_NOUNS)]}"
        if index > 20:
            title = f"{title} {1 + ((index - 1) // 20)}"

        description = rng.choice(DESCRIPTION_TEMPLATES).format(
            tone=rng.choice(TONES),
            subject=rng.choice(SUBJECTS),
            conflict=rng.choice(CONFLICTS),
            event=rng.choice(EVENTS),
            secret=rng.choice(SECRETS),
        )

        genre = f"{primary}|{secondary}"
        embedding = vector_from_genres([primary, secondary], f"movie-vector:{index}")

        commercial_bias = {
            "Action": 170,
            "Adventure": 155,
            "Animation": 145,
            "Sci-Fi": 160,
            "Fantasy": 135,
            "Comedy": 90,
            "Romance": 65,
            "Horror": 80,
            "Thriller": 100,
            "Crime": 70,
            "Drama": 55,
            "Family": 125,
            "Documentary": 28,
            "Historical": 45,
            "Music": 60,
            "Sports": 72,
        }
        revenue = commercial_bias[primary] + commercial_bias[secondary] + rng.uniform(-35, 165)
        if index in {9, 24, 38, 57, 73, 89} or index % 137 == 0:
            revenue += 420
        revenue = round(max(12.0, min(revenue, 980.0)), 1)
        score = round(5.8 + rng.random() * 3.8 + (0.35 if revenue > 500 else 0), 1)
        score = min(score, 9.8)

        movies.append(
            Movie(
                key=f"{MOVIE_PREFIX}{index:03d}",
                movie_id=f"M{index:03d}",
                title=title,
                genre=genre,
                description=description,
                revenue=revenue,
                score=score,
                embedding=embedding,
            )
        )

    return movies


def create_users(movies: list[Movie]) -> list[User]:
    users = []
    for key, name, genres, preferences in USER_PERSONAS:
        user_id = key.split(":", 1)[1]
        embedding = vector_from_genres(genres, f"user-vector:{user_id}", noise=0.02)
        ranked_movies = sorted(
            movies,
            key=lambda movie: (
                cosine_similarity(embedding, movie.embedding),
                movie.score,
                movie.revenue,
            ),
            reverse=True,
        )
        watch_history = [movie.movie_id for movie in ranked_movies[3:8]]
        users.append(User(key, user_id, name, watch_history, preferences, embedding))
    return users


def drop_index(client: redis.Redis, index_name: str) -> None:
    try:
        client.execute_command("FT.DROPINDEX", index_name, "DD")
    except ResponseError as exc:
        if "not found" not in str(exc).lower():
            raise


def delete_prefix(client: redis.Redis, prefix: str, batch_size: int = 1000) -> None:
    batch = []
    for key in client.scan_iter(f"{prefix}*", count=batch_size):
        batch.append(key)
        if len(batch) >= batch_size:
            client.delete(*batch)
            batch.clear()
    if batch:
        client.delete(*batch)


def index_value(info: object, field: bytes) -> object:
    if isinstance(info, dict):
        return info.get(field)
    if isinstance(info, list):
        for index in range(0, len(info), 2):
            if info[index] == field:
                return info[index + 1]
    return None


def wait_for_index(client: redis.Redis, index_name: str, expected_docs: int, timeout_seconds: float = 10.0) -> None:
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        info = client.execute_command("FT.INFO", index_name)
        num_docs = int(index_value(info, b"num_docs") or 0)
        indexing = int(index_value(info, b"indexing") or 0)
        failures = int(index_value(info, b"hash_indexing_failures") or 0)
        if failures:
            raise RuntimeError(f"{index_name} has {failures} hash indexing failures")
        if num_docs >= expected_docs and indexing == 0:
            return
        time.sleep(0.05)
    raise TimeoutError(f"{index_name} did not finish indexing {expected_docs} docs within {timeout_seconds:.1f}s")


def index_ready(client: redis.Redis, index_name: str, expected_docs: int) -> bool:
    try:
        info = client.execute_command("FT.INFO", index_name)
    except ResponseError as exc:
        if "not found" in str(exc).lower():
            return False
        raise

    num_docs = int(index_value(info, b"num_docs") or 0)
    indexing = int(index_value(info, b"indexing") or 0)
    failures = int(index_value(info, b"hash_indexing_failures") or 0)
    return num_docs >= expected_docs and indexing == 0 and failures == 0


def create_movie_index(client: redis.Redis, index_name: str, vector_algorithm: str, vector_attrs: list[str]) -> None:
    client.execute_command(
        "FT.CREATE",
        index_name,
        "ON",
        "HASH",
        "PREFIX",
        "1",
        MOVIE_PREFIX,
        "SCHEMA",
        "movieId",
        "TAG",
        "SORTABLE",
        "title",
        "TEXT",
        "genre",
        "TAG",
        "SORTABLE",
        "description",
        "TEXT",
        "revenue",
        "NUMERIC",
        "SORTABLE",
        "score",
        "NUMERIC",
        "SORTABLE",
        "embedding",
        "VECTOR",
        vector_algorithm,
        str(len(vector_attrs)),
        *vector_attrs,
    )


def create_indexes(client: redis.Redis) -> None:
    drop_index(client, LEGACY_MOVIE_HNSW_INDEX)
    drop_index(client, MOVIE_INDEX)
    drop_index(client, USER_INDEX)
    delete_prefix(client, MOVIE_PREFIX)
    delete_prefix(client, USER_PREFIX)
    create_movie_index(
        client,
        MOVIE_INDEX,
        "FLAT",
        ["TYPE", "FLOAT32", "DIM", str(DIMENSIONS), "DISTANCE_METRIC", "COSINE"],
    )
    client.execute_command(
        "FT.CREATE",
        USER_INDEX,
        "ON",
        "HASH",
        "PREFIX",
        "1",
        USER_PREFIX,
        "SCHEMA",
        "userId",
        "TAG",
        "SORTABLE",
        "name",
        "TEXT",
        "watchHistory",
        "TEXT",
        "preferences",
        "TEXT",
        "embedding",
        "VECTOR",
        "FLAT",
        "6",
        "TYPE",
        "FLOAT32",
        "DIM",
        str(DIMENSIONS),
        "DISTANCE_METRIC",
        "COSINE",
    )


def seed(client: redis.Redis) -> tuple[list[Movie], list[User]]:
    movies = create_movies()
    users = create_users(movies)
    create_indexes(client)

    pipeline = client.pipeline(transaction=False)
    for count, movie in enumerate(movies, start=1):
        pipeline.execute_command(
            "HSET",
            movie.key,
            "movieId",
            movie.movie_id,
            "title",
            movie.title,
            "genre",
            movie.genre,
            "description",
            movie.description,
            "revenue",
            f"{movie.revenue:.1f}",
            "score",
            f"{movie.score:.1f}",
            "embedding",
            pack_vector(movie.embedding),
        )
        if count % 500 == 0:
            pipeline.execute()

    for user in users:
        pipeline.execute_command(
            "HSET",
            user.key,
            "userId",
            user.user_id,
            "name",
            user.name,
            "watchHistory",
            ",".join(user.watch_history),
            "preferences",
            user.preferences,
            "embedding",
            pack_vector(user.embedding),
        )
    pipeline.execute()
    wait_for_index(client, MOVIE_INDEX, len(movies))
    wait_for_index(client, USER_INDEX, len(users))

    return movies, users


def load_users(client: redis.Redis) -> list[User]:
    users = []
    for key, name, _genres, preferences in USER_PERSONAS:
        data = client.hgetall(key)
        if not data or b"embedding" not in data:
            raise RuntimeError(f"Missing existing user vector at {key}; run with --reseed to rebuild the demo dataset")

        users.append(
            User(
                key=key,
                user_id=decode(data.get(b"userId", key.split(":", 1)[1])),
                name=decode(data.get(b"name", name)),
                watch_history=decode(data.get(b"watchHistory", b"")).split(","),
                preferences=decode(data.get(b"preferences", preferences)),
                embedding=unpack_vector(data[b"embedding"]),
            )
        )
    return users


def ensure_dataset(client: redis.Redis, reseed: bool) -> list[User]:
    if reseed or not (
        index_ready(client, MOVIE_INDEX, MOVIE_COUNT)
        and index_ready(client, USER_INDEX, USER_COUNT)
        and client.exists(f"{MOVIE_PREFIX}001")
        and client.exists(f"{USER_PREFIX}001")
    ):
        _, users = seed(client)
        return users

    return load_users(client)


def recommendation_expression(revenue_weight: float, rating_weight: float) -> tuple[str, str, str]:
    semantic_weight = 1.0 - revenue_weight - rating_weight
    if semantic_weight <= 0:
        raise ValueError("revenue_weight + rating_weight must be less than 1.0")

    semantic_expr = "(1/(1+@vector_distance))"
    revenue_expr = "(@revenue/1000)"
    boosted_expr = (
        f"((@semantic_score*{semantic_weight:.4f})"
        f"+(@revenue_score*{revenue_weight:.4f})"
        f"+((@score/10)*{rating_weight:.4f}))"
    )
    return semantic_expr, revenue_expr, boosted_expr


def recommend_movies(
    client: redis.Redis,
    user: User,
    top_k: int,
    candidate_pool: int,
    revenue_weight: float,
    rating_weight: float,
) -> tuple[list[dict[str, str]], float]:
    semantic_expr, revenue_expr, boosted_expr = recommendation_expression(revenue_weight, rating_weight)
    started = time.perf_counter()
    response = client.execute_command(
        "FT.AGGREGATE",
        MOVIE_INDEX,
        f"*=>[KNN {candidate_pool} @embedding $user_vector AS vector_distance]",
        "PARAMS",
        "2",
        "user_vector",
        pack_vector(user.embedding),
        "LOAD",
        "7",
        "@movieId",
        "@title",
        "@genre",
        "@description",
        "@revenue",
        "@score",
        "@vector_distance",
        "APPLY",
        semantic_expr,
        "AS",
        "semantic_score",
        "APPLY",
        revenue_expr,
        "AS",
        "revenue_score",
        "APPLY",
        boosted_expr,
        "AS",
        "boosted_score",
        "SORTBY",
        "2",
        "@boosted_score",
        "DESC",
        "LIMIT",
        "0",
        str(top_k),
        "DIALECT",
        "2",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return parse_aggregate(response), elapsed_ms


def parse_aggregate(response: object) -> list[dict[str, str]]:
    if isinstance(response, dict):
        rows = []
        for result in response.get(b"results", []):
            if not isinstance(result, dict):
                continue
            attributes = result.get(b"extra_attributes", {})
            rows.append({decode(key): decode(value) for key, value in attributes.items()})
        return rows

    if not isinstance(response, list) or not response:
        return []

    rows = []
    for row in response[1:]:
        if not isinstance(row, list):
            continue
        parsed = {}
        for index in range(0, len(row), 2):
            parsed[decode(row[index])] = decode(row[index + 1])
        rows.append(parsed)
    return rows


def print_rule(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_command_pattern(candidate_pool: int, top_k: int, revenue_weight: float, rating_weight: float) -> None:
    semantic_expr, revenue_expr, boosted_expr = recommendation_expression(revenue_weight, rating_weight)
    print_rule("One Redis Command For Recommendations")
    print("The app passes the selected user's binary vector as $user_vector; Redis computes and sorts the boost.")
    print()
    print(
        f"FT.AGGREGATE {MOVIE_INDEX} "
        f"\"*=>[KNN {candidate_pool} @embedding $user_vector AS vector_distance]\" "
        "PARAMS 2 user_vector <binary FLOAT32 user vector> "
        "LOAD 7 @movieId @title @genre @description @revenue @score @vector_distance "
        f"APPLY \"{semantic_expr}\" AS semantic_score "
        f"APPLY \"{revenue_expr}\" AS revenue_score "
        f"APPLY \"{boosted_expr}\" AS boosted_score "
        f"SORTBY 2 @boosted_score DESC LIMIT 0 {top_k} DIALECT 2"
    )


def fmt_float(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def print_recommendations(
    client: redis.Redis,
    users: list[User],
    top_k: int,
    candidate_pool: int,
    revenue_weight: float,
    rating_weight: float,
    selected_user: str | None,
) -> None:
    print_rule("Recommendations")
    print(
        f"Boost formula: semantic*(1 - revenue_weight - rating_weight) + revenue_norm*{revenue_weight:.2f} "
        f"+ rating_norm*{rating_weight:.2f}"
    )
    print("semantic = 1/(1 + vector_distance), revenue_norm = revenue_millions/1000, rating_norm = score/10")

    selected = [
        user
        for user in users
        if selected_user is None or selected_user in {user.key, user.user_id, user.name.lower(), user.name}
    ]
    if not selected:
        raise ValueError(f"No generated user matched {selected_user!r}")

    for user in selected:
        rows, elapsed_ms = recommend_movies(client, user, top_k, candidate_pool, revenue_weight, rating_weight)
        print()
        print(f"{user.key} {user.name}")
        print(f"Likes:   {user.preferences}")
        print(f"Watched: {', '.join(user.watch_history)}")
        print(f"FT.AGGREGATE latency: {elapsed_ms:.3f} ms")
        print("-" * 118)
        print(f"{'#':<3} {'movieId':<7} {'title':<28} {'genre':<22} {'distance':>9} {'revenue':>10} {'score':>6} {'boosted':>9}")
        print("-" * 118)
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank:<3} "
                f"{row['movieId']:<7} "
                f"{row['title'][:28]:<28} "
                f"{row['genre'][:22]:<22} "
                f"{fmt_float(row['vector_distance']):>9} "
                f"${float(row['revenue']):>8.1f}M "
                f"{float(row['score']):>6.1f} "
                f"{fmt_float(row['boosted_score']):>9}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Redis vector search demo with revenue-boosted recommendations.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-pool", type=int, default=MOVIE_COUNT)
    parser.add_argument("--revenue-weight", type=float, default=0.25)
    parser.add_argument("--rating-weight", type=float, default=0.10)
    parser.add_argument("--reseed", action="store_true", help="Rebuild the movies:/users: demo dataset and indexes before querying.")
    parser.add_argument("--user", help="Optional user key/id/name to show one recommendation set, e.g. users:001 or Maya.")
    args = parser.parse_args()

    if args.candidate_pool < args.top_k:
        raise SystemExit("--candidate-pool must be >= --top-k")
    if args.candidate_pool > MOVIE_COUNT:
        raise SystemExit(f"--candidate-pool must be <= {MOVIE_COUNT} for this demo dataset")
    if args.revenue_weight < 0 or args.rating_weight < 0:
        raise SystemExit("--revenue-weight and --rating-weight must be >= 0")
    if args.revenue_weight + args.rating_weight >= 1.0:
        raise SystemExit("--revenue-weight + --rating-weight must be less than 1.0")
    client = redis.Redis(host=args.host, port=args.port, decode_responses=False)
    client.ping()
    users = ensure_dataset(client, args.reseed)
    print_command_pattern(args.candidate_pool, args.top_k, args.revenue_weight, args.rating_weight)
    print_recommendations(
        client,
        users,
        args.top_k,
        args.candidate_pool,
        args.revenue_weight,
        args.rating_weight,
        args.user,
    )


if __name__ == "__main__":
    main()
