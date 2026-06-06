#!/usr/bin/env python3
"""
Seed a larger movie-vector dataset into Redis.

Default target:
  - 250,000 movie hashes
  - keys under movies_full:
  - same fields as the demo movie hashes:
    movieId, title, genre, description, revenue, score, embedding
"""

from __future__ import annotations

import argparse
import time

import redis

from demo import (
    CONFLICTS,
    DESCRIPTION_TEMPLATES,
    DIMENSIONS,
    EVENTS,
    GENRE_PAIRINGS,
    MOVIE_ADJECTIVES,
    MOVIE_NOUNS,
    SECRETS,
    SUBJECTS,
    TONES,
    pack_vector,
    stable_rng,
    vector_from_genres,
)


MOVIES_FULL_PREFIX = "movies_full:"
DEFAULT_MOVIE_COUNT = 250_000


COMMERCIAL_BIAS = {
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


def delete_prefix(client: redis.Redis, prefix: str, batch_size: int) -> int:
    deleted = 0
    batch = []
    for key in client.scan_iter(f"{prefix}*", count=batch_size):
        batch.append(key)
        if len(batch) >= batch_size:
            deleted += client.delete(*batch)
            batch.clear()
    if batch:
        deleted += client.delete(*batch)
    return deleted


def count_prefix(client: redis.Redis, prefix: str, batch_size: int) -> int:
    return sum(1 for _ in client.scan_iter(f"{prefix}*", count=batch_size))


def movie_record(index: int) -> tuple[str, tuple[object, ...]]:
    primary, secondary = GENRE_PAIRINGS[(index - 1) % len(GENRE_PAIRINGS)]
    if index % 7 == 0:
        primary, secondary = secondary, primary

    rng = stable_rng(f"movie-full:{index}")
    title = f"{MOVIE_ADJECTIVES[(index - 1) % len(MOVIE_ADJECTIVES)]} {MOVIE_NOUNS[(index * 7) % len(MOVIE_NOUNS)]}"
    title = f"{title} {1 + ((index - 1) // len(MOVIE_NOUNS))}"

    description = rng.choice(DESCRIPTION_TEMPLATES).format(
        tone=rng.choice(TONES),
        subject=rng.choice(SUBJECTS),
        conflict=rng.choice(CONFLICTS),
        event=rng.choice(EVENTS),
        secret=rng.choice(SECRETS),
    )

    revenue = COMMERCIAL_BIAS[primary] + COMMERCIAL_BIAS[secondary] + rng.uniform(-35, 165)
    if index % 137 == 0 or index % 997 == 0:
        revenue += 420
    revenue = round(max(12.0, min(revenue, 980.0)), 1)
    score = round(5.8 + rng.random() * 3.8 + (0.35 if revenue > 500 else 0), 1)
    score = min(score, 9.8)

    key = f"{MOVIES_FULL_PREFIX}{index:06d}"
    movie_id = f"MF{index:06d}"
    genre = f"{primary}|{secondary}"
    embedding = vector_from_genres([primary, secondary], f"movie-full-vector:{index}")

    return key, (
        "movieId",
        movie_id,
        "title",
        title,
        "genre",
        genre,
        "description",
        description,
        "revenue",
        f"{revenue:.1f}",
        "score",
        f"{score:.1f}",
        "embedding",
        pack_vector(embedding),
    )


def seed_movies_full(client: redis.Redis, count: int, batch_size: int, replace: bool) -> None:
    if replace:
        started_delete = time.perf_counter()
        deleted = delete_prefix(client, MOVIES_FULL_PREFIX, batch_size)
        print(f"Deleted {deleted:,} existing {MOVIES_FULL_PREFIX} keys in {time.perf_counter() - started_delete:.2f}s")

    started = time.perf_counter()
    pipeline = client.pipeline(transaction=False)
    for index in range(1, count + 1):
        key, fields = movie_record(index)
        pipeline.hset(key, mapping=dict(zip(fields[0::2], fields[1::2])))
        if index % batch_size == 0:
            pipeline.execute()
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0
            print(f"Inserted {index:,}/{count:,} movies ({rate:,.0f}/s)")

    remaining = count % batch_size
    if remaining:
        pipeline.execute()

    elapsed = time.perf_counter() - started
    print(f"Inserted {count:,} {MOVIES_FULL_PREFIX} movie vectors in {elapsed:.2f}s ({count / elapsed:,.0f}/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed 250,000 movie vector hashes under movies_full:.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--count", type=int, default=DEFAULT_MOVIE_COUNT)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--no-replace", action="store_true", help="Do not delete existing movies_full:* keys before inserting.")
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    client = redis.Redis(host=args.host, port=args.port, decode_responses=False)
    client.ping()

    print(f"Target namespace: {MOVIES_FULL_PREFIX}")
    print(f"Movie count:      {args.count:,}")
    print(f"Vector shape:     {DIMENSIONS}D FLOAT32")
    seed_movies_full(client, args.count, args.batch_size, replace=not args.no_replace)
    verified = count_prefix(client, MOVIES_FULL_PREFIX, args.batch_size)
    print(f"Verified keys:    {verified:,} matching {MOVIES_FULL_PREFIX}*")

    sample = client.hgetall(f"{MOVIES_FULL_PREFIX}000001")
    print("Sample:")
    print(f"  key={MOVIES_FULL_PREFIX}000001")
    print(f"  movieId={sample[b'movieId'].decode()} genre={sample[b'genre'].decode()} revenue=${float(sample[b'revenue']):.1f}M score={float(sample[b'score']):.1f}")
    print(f"  title={sample[b'title'].decode()}")


if __name__ == "__main__":
    main()
