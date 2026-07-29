#!/usr/bin/env python3
"""Score PKU-SafeRLHF response Parquets for joint-distribution plots.

The generated cache is deliberately tied to the exact source Parquet and to
each prompt/response pair with SHA-256 hashes.  This prevents an old score
cache from being combined with newly generated responses, which was possible
with the original row-order-only JSONL files.

Examples
--------
Score all Qwen GRPO/GDPO/HVPO artifacts::

    python playground/visualization/record_score.py

Create a deterministic 1,000-response preview for every artifact::

    python playground/visualization/record_score.py --limit 1000 --seed 42

The helpfulness and harmlessness gRPC reward services must already be running.
Interrupted runs can be resumed because completed rows are appended to the
validated cache after every batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = 2
DEFAULT_EXPERIMENTS = (
    "qwen2.5-1.5b_grpo",
    "qwen2.5-1.5b_gdpo",
    "qwen2.5-1.5b_hvpo",
    "qwen2.5-3b_grpo",
    "qwen2.5-3b_gdpo",
    "qwen2.5-3b_hvpo",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first_response(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("row does not contain a generated response")
    response = value[0]
    if not isinstance(response, str) or not response.strip():
        raise ValueError("first generated response is empty")
    return response


def _question(extra_info: Any) -> tuple[str, Any]:
    if not isinstance(extra_info, dict):
        raise ValueError("extra_info must be a dictionary")
    question = extra_info.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("extra_info.question is missing")
    dataset_index = extra_info.get("index")
    if isinstance(dataset_index, np.generic):
        dataset_index = dataset_index.item()
    return question, dataset_index


def _selected_rows(size: int, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit <= 0 or limit >= size:
        return np.arange(size, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=limit, replace=False))


def load_jobs(
    parquet_path: Path,
    experiment: str,
    limit: int | None,
    seed: int,
) -> tuple[str, list[dict[str, Any]]]:
    source_sha256 = sha256_file(parquet_path)
    frame = pd.read_parquet(parquet_path, columns=["extra_info", "responses"])
    positions = _selected_rows(len(frame), limit, seed)
    jobs: list[dict[str, Any]] = []
    for source_row in positions:
        row = frame.iloc[int(source_row)]
        question, dataset_index = _question(row["extra_info"])
        response = _first_response(row["responses"])
        jobs.append(
            {
                "schema_version": SCHEMA_VERSION,
                "experiment": experiment,
                "source_parquet": parquet_path.name,
                "source_sha256": source_sha256,
                "source_row": int(source_row),
                "dataset_index": dataset_index,
                "question": question,
                "response": response,
                "question_sha256": sha256_text(question),
                "response_sha256": sha256_text(response),
            }
        )
    return source_sha256, jobs


def load_cache(
    cache_path: Path,
    source_sha256: str,
    jobs_by_row: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not cache_path.exists():
        return {}

    cached: dict[int, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{cache_path}:{line_number} is not valid JSON; "
                    "remove it or rerun with --overwrite"
                ) from exc

            if record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"{cache_path} uses the legacy score-cache schema; "
                    "rerun with --overwrite"
                )
            if record.get("source_sha256") != source_sha256:
                raise ValueError(
                    f"{cache_path} was produced from a different Parquet; "
                    "rerun with --overwrite"
                )

            source_row = int(record["source_row"])
            job = jobs_by_row.get(source_row)
            if job is not None and (
                record.get("question_sha256") != job["question_sha256"]
                or record.get("response_sha256") != job["response_sha256"]
            ):
                raise ValueError(
                    f"{cache_path}:{line_number} does not match source row "
                    f"{source_row}; rerun with --overwrite"
                )
            cached[source_row] = record
    return cached


def _score_task(task: tuple[str, str, str, str, str]) -> float:
    objective, question, response, host, port = task
    # Import lazily so cache inspection and --help work without grpc installed.
    from recipe.amo_safe.reward_client import compute_reward_score

    score = compute_reward_score(question, response, host, port)
    if not np.isfinite(score):
        raise ValueError(f"{objective} reward service returned {score!r}")
    return float(score)


def _task_pairs(
    jobs: Iterable[dict[str, Any]],
    host: str,
    helpful_port: str,
    harmless_port: str,
) -> list[tuple[str, str, str, str, str]]:
    tasks: list[tuple[str, str, str, str, str]] = []
    for job in jobs:
        tasks.append(("helpfulness", job["question"], job["response"], host, helpful_port))
        tasks.append(("harmlessness", job["question"], job["response"], host, harmless_port))
    return tasks


def score_experiment(
    experiment: str,
    result_dir: Path,
    score_dir: Path,
    limit: int | None,
    seed: int,
    host: str,
    helpful_port: str,
    harmless_port: str,
    workers: int,
    batch_size: int,
    overwrite: bool,
) -> Path:
    parquet_path = result_dir / f"{experiment}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"missing result Parquet: {parquet_path}")

    score_dir.mkdir(parents=True, exist_ok=True)
    cache_path = score_dir / f"{experiment}.jsonl"
    if overwrite and cache_path.exists():
        cache_path.unlink()

    source_sha256, jobs = load_jobs(parquet_path, experiment, limit, seed)
    jobs_by_row = {job["source_row"]: job for job in jobs}
    cached = load_cache(cache_path, source_sha256, jobs_by_row)
    missing = [job for job in jobs if job["source_row"] not in cached]

    print(
        f"[{experiment}] selected={len(jobs)} cached={len(jobs) - len(missing)} "
        f"missing={len(missing)} source={source_sha256[:12]}"
    )
    if not missing:
        return cache_path

    completed = 0
    with ThreadPoolExecutor(max_workers=max(2, workers)) as executor:
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            tasks = _task_pairs(batch, host, helpful_port, harmless_port)
            scores = list(executor.map(_score_task, tasks))

            with cache_path.open("a", encoding="utf-8") as stream:
                for offset, job in enumerate(batch):
                    record = {key: value for key, value in job.items() if key not in {"question", "response"}}
                    record["helpful_score"] = scores[2 * offset]
                    record["harmless_score"] = scores[2 * offset + 1]
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()

            completed += len(batch)
            print(f"[{experiment}] scored {completed}/{len(missing)} new rows", flush=True)

    return cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score PKU-SafeRLHF response Parquets with validated caches."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        dest="experiments",
        action="append",
        help="experiment stem; repeat to select multiple (default: six Qwen runs)",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/PKU-SafeRLHF"),
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=Path("playground/visualization/scored_responses"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="deterministic sample size per run; omit or use 0 for all rows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--helpful-port", default="50051")
    parser.add_argument("--harmless-port", default="50052")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing cache after an explicit source/protocol change",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = tuple(args.experiments or DEFAULT_EXPERIMENTS)
    for experiment in experiments:
        score_experiment(
            experiment=experiment,
            result_dir=args.result_dir,
            score_dir=args.score_dir,
            limit=args.limit,
            seed=args.seed,
            host=args.host,
            helpful_port=args.helpful_port,
            harmless_port=args.harmless_port,
            workers=args.workers,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
