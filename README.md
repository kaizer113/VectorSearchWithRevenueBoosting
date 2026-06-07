# Redis Vector Search With Revenue Boosting

Customer demo showing how Redis Query Engine can combine vector similarity with business signals in a single recommendation query.

The demo creates movie and user vectors, then recommends movies for each user with one `FT.AGGREGATE` command. Redis computes vector distance, normalizes revenue, blends the scores, and sorts by a tunable boosted relevance score.

## What It Demonstrates

- Vector search over movie embeddings stored in Redis hashes
- User-to-movie recommendations using `KNN`
- Revenue-aware re-ranking inside Redis with `APPLY`
- Query latency measurements around the actual `FT.AGGREGATE` command
- A larger `250,000` movie dataset using HNSW for approximate vector search

## Requirements

- Redis 8.8 running locally on port `6379`
- Python 3
- `redis-py`

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 5,000 Movie Demo

Run the main demo:

```bash
.venv/bin/python demo.py
```

By default, `demo.py` reuses the existing `movies:` and `users:` data when the Redis indexes are healthy. Rebuild the 5,000 movie demo dataset only when needed:

```bash
.venv/bin/python demo.py --reseed
```

Run a focused recommendation for Maya:

```bash
.venv/bin/python demo.py --user users:001 --candidate-pool 250 --revenue-weight 0.10 --rating-weight 0.10
```

The demo seeds:

- `5,000` movie hashes under `movies:`
- `10` user hashes under `users:`
- `idx:movies` exact `FLAT` vector index
- `idx:users` user vector index

## One Redis Command

The recommendation query passes the selected user's binary vector into Redis and asks Redis to compute the boosted ranking:

```text
FT.AGGREGATE idx:movies "*=>[KNN 5000 @embedding $user_vector AS vector_distance]"
  PARAMS 2 user_vector <binary FLOAT32 user vector>
  LOAD 7 @movieId @title @genre @description @revenue @score @vector_distance
  APPLY "(1/(1+@vector_distance))" AS semantic_score
  APPLY "(@revenue/1000)" AS revenue_score
  APPLY "((@semantic_score*0.6500)+(@revenue_score*0.2500)+((@score/10)*0.1000))" AS boosted_score
  SORTBY 2 @boosted_score DESC
  LIMIT 0 3
  DIALECT 2
```

Lower `vector_distance` means a closer semantic match. The demo converts that distance into `semantic_score`, then blends it with normalized revenue and movie rating:

```text
boosted_score =
  semantic_score * (1 - revenue_weight - rating_weight)
  + revenue_score * revenue_weight
  + rating_score * rating_weight
```

Tune the business boost:

```bash
.venv/bin/python demo.py --revenue-weight 0.35 --rating-weight 0.10 --user users:001
```

## Large Dataset Demo

Seed a larger catalog under `movies_full:`:

```bash
.venv/bin/python seed_movies_full.py
```

Defaults:

- `250,000` movie hashes
- `movies_full:<id>` key namespace
- Fields: `movieId`, `title`, `genre`, `description`, `revenue`, `score`, `embedding`
- `16D FLOAT32` vectors

Run recommendations against the large dataset:

```bash
.venv/bin/python recommend_movies_full.py --user users:001 --candidate-pool 1000 --revenue-weight 0.10 --rating-weight 0.10
```

This creates or reuses `idx:movies_full_hnsw`, an HNSW vector index over `movies_full:`.

To rebuild only the large HNSW index without deleting the `movies_full:*` hashes:

```bash
.venv/bin/python recommend_movies_full.py --recreate-index
```

## Candidate Pool

`--candidate-pool` controls the `KNN` value.

For example, `--candidate-pool 1000` means Redis retrieves the 1,000 nearest vector candidates, then applies the revenue/rating boost and returns the top 3. Larger pools allow more business-aware re-ranking but cost more latency.
