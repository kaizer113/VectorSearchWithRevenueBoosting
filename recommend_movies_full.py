#!/usr/bin/env python3
"""
Run revenue-boosted recommendations against the large movies_full: dataset.

Uses one Redis command for the recommendation step:
  FT.AGGREGATE idx:movies_full_hnsw ... KNN ... APPLY boosted_score ... SORTBY ...
"""

from __future__ import annotations

import argparse
import time

import redis
from redis.exceptions import ResponseError

from demo import (
    DIMENSIONS,
    USER_PERSONAS,
    User,
    fmt_float,
    index_value,
    pack_vector,
    parse_aggregate,
    recommendation_expression,
    vector_from_genres,
)
from seed_movies_full import MOVIES_FULL_PREFIX


MOVIES_FULL_HNSW_INDEX = "idx:movies_full_hnsw"
DEFAULT_CANDIDATE_POOL = 1_000


def index_exists(client: redis.Redis, index_name: str) -> bool:
    try:
        client.execute_command("FT.INFO", index_name)
        return True
    except ResponseError as exc:
        if "not found" in str(exc).lower():
            return False
        raise


def count_prefix(client: redis.Redis, prefix: str, batch_size: int = 10_000) -> int:
    return sum(1 for _ in client.scan_iter(f"{prefix}*", count=batch_size))


def create_movies_full_hnsw_index(
    client: redis.Redis,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_runtime: int,
    recreate: bool,
) -> None:
    if recreate and index_exists(client, MOVIES_FULL_HNSW_INDEX):
        client.execute_command("FT.DROPINDEX", MOVIES_FULL_HNSW_INDEX)

    if index_exists(client, MOVIES_FULL_HNSW_INDEX):
        return

    client.execute_command(
        "FT.CREATE",
        MOVIES_FULL_HNSW_INDEX,
        "ON",
        "HASH",
        "PREFIX",
        "1",
        MOVIES_FULL_PREFIX,
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
        "HNSW",
        "12",
        "TYPE",
        "FLOAT32",
        "DIM",
        str(DIMENSIONS),
        "DISTANCE_METRIC",
        "COSINE",
        "M",
        str(hnsw_m),
        "EF_CONSTRUCTION",
        str(hnsw_ef_construction),
        "EF_RUNTIME",
        str(hnsw_ef_runtime),
    )


def wait_for_index(client: redis.Redis, index_name: str, expected_docs: int, timeout_seconds: float) -> None:
    started = time.perf_counter()
    last_report = 0.0
    while True:
        info = client.execute_command("FT.INFO", index_name)
        num_docs = int(index_value(info, b"num_docs") or 0)
        indexing = int(index_value(info, b"indexing") or 0)
        failures = int(index_value(info, b"hash_indexing_failures") or 0)
        elapsed = time.perf_counter() - started

        if failures:
            raise RuntimeError(f"{index_name} has {failures} hash indexing failures")
        if num_docs >= expected_docs and indexing == 0:
            print(f"{index_name} ready: {num_docs:,} docs indexed in {elapsed:.2f}s")
            return
        if elapsed - last_report >= 2.0:
            print(f"{index_name} indexing: {num_docs:,}/{expected_docs:,} docs, indexing={indexing}")
            last_report = elapsed
        if elapsed > timeout_seconds:
            raise TimeoutError(f"{index_name} did not index {expected_docs:,} docs within {timeout_seconds:.1f}s")
        time.sleep(0.25)


def select_user(selected_user: str | None) -> User:
    for key, name, genres, preferences in USER_PERSONAS:
        user_id = key.split(":", 1)[1]
        if selected_user is None or selected_user in {key, user_id, name, name.lower()}:
            embedding = vector_from_genres(genres, f"user-vector:{user_id}", noise=0.02)
            return User(key, user_id, name, ["large catalog demo"], preferences, embedding)
    raise ValueError(f"No generated user matched {selected_user!r}")


def recommend_movies_full(
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
        MOVIES_FULL_HNSW_INDEX,
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


def print_command(candidate_pool: int, top_k: int, revenue_weight: float, rating_weight: float) -> None:
    semantic_expr, revenue_expr, boosted_expr = recommendation_expression(revenue_weight, rating_weight)
    print()
    print("One Redis Command")
    print("=================")
    print(
        f"FT.AGGREGATE {MOVIES_FULL_HNSW_INDEX} "
        f"\"*=>[KNN {candidate_pool} @embedding $user_vector AS vector_distance]\" "
        "PARAMS 2 user_vector <binary FLOAT32 user vector> "
        "LOAD 7 @movieId @title @genre @description @revenue @score @vector_distance "
        f"APPLY \"{semantic_expr}\" AS semantic_score "
        f"APPLY \"{revenue_expr}\" AS revenue_score "
        f"APPLY \"{boosted_expr}\" AS boosted_score "
        f"SORTBY 2 @boosted_score DESC LIMIT 0 {top_k} DIALECT 2"
    )


def print_recommendations(user: User, rows: list[dict[str, str]], elapsed_ms: float) -> None:
    print()
    print(f"{user.key} {user.name}")
    print(f"Likes:   {user.preferences}")
    print(f"Dataset: {MOVIES_FULL_PREFIX} via {MOVIES_FULL_HNSW_INDEX}")
    print(f"FT.AGGREGATE latency: {elapsed_ms:.3f} ms")
    print("-" * 122)
    print(f"{'#':<3} {'movieId':<9} {'title':<30} {'genre':<22} {'distance':>9} {'revenue':>10} {'score':>6} {'boosted':>9}")
    print("-" * 122)
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:<3} "
            f"{row['movieId']:<9} "
            f"{row['title'][:30]:<30} "
            f"{row['genre'][:22]:<22} "
            f"{fmt_float(row['vector_distance']):>9} "
            f"${float(row['revenue']):>8.1f}M "
            f"{float(row['score']):>6.1f} "
            f"{fmt_float(row['boosted_score']):>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend movies from the 250k movies_full: dataset.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--revenue-weight", type=float, default=0.10)
    parser.add_argument("--rating-weight", type=float, default=0.10)
    parser.add_argument("--user", default="users:001", help="User key/id/name, e.g. users:001 or Maya.")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-runtime", type=int, default=100)
    parser.add_argument("--recreate-index", action="store_true", help="Drop and rebuild idx:movies_full_hnsw without deleting movies_full:* data.")
    parser.add_argument("--index-timeout", type=float, default=180.0)
    args = parser.parse_args()

    if args.candidate_pool < args.top_k:
        raise SystemExit("--candidate-pool must be >= --top-k")
    if args.revenue_weight < 0 or args.rating_weight < 0:
        raise SystemExit("--revenue-weight and --rating-weight must be >= 0")
    if args.revenue_weight + args.rating_weight >= 1.0:
        raise SystemExit("--revenue-weight + --rating-weight must be less than 1.0")
    if args.hnsw_m <= 0 or args.hnsw_ef_construction <= 0 or args.hnsw_ef_runtime <= 0:
        raise SystemExit("HNSW settings must be positive integers")

    client = redis.Redis(host=args.host, port=args.port, decode_responses=False)
    client.ping()

    total_movies = count_prefix(client, MOVIES_FULL_PREFIX)
    if total_movies == 0:
        raise SystemExit(f"No {MOVIES_FULL_PREFIX} keys found. Run: .venv/bin/python seed_movies_full.py")

    print(f"Large movie dataset: {total_movies:,} keys under {MOVIES_FULL_PREFIX}")
    create_movies_full_hnsw_index(
        client,
        args.hnsw_m,
        args.hnsw_ef_construction,
        args.hnsw_ef_runtime,
        args.recreate_index,
    )
    wait_for_index(client, MOVIES_FULL_HNSW_INDEX, total_movies, args.index_timeout)

    user = select_user(args.user)
    print_command(args.candidate_pool, args.top_k, args.revenue_weight, args.rating_weight)
    rows, elapsed_ms = recommend_movies_full(
        client,
        user,
        args.top_k,
        args.candidate_pool,
        args.revenue_weight,
        args.rating_weight,
    )
    print_recommendations(user, rows, elapsed_ms)


if __name__ == "__main__":
    main()
