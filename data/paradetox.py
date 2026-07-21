# Copyright 2025 Rihong Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocess ParaDetox into leakage-safe train/test parquet files.

ParaDetox (s-nlp/paradetox) is a parallel text detoxification corpus. A toxic
source may occur in multiple rows with different neutral rewrites. Splitting
rows independently therefore leaks identical RL prompts between train and test.

By default this script normalizes each toxic source, assigns the complete source
group to one split, and retains one deterministic row per source. Use
--keep_duplicate_prompts to retain all references in the leakage-safe grouped
split, or --legacy_row_split to reproduce the former row-level split.

The three reward axes are style transfer accuracy (STA), content preservation
(SIM), and fluency (FL). The toxic source is stored in extra_info because SIM
uses it as the comparison target.
"""

import argparse
import hashlib
import os
import unicodedata
from dataclasses import dataclass
from typing import Sequence


DETOX_INSTRUCTION = (
    "Rewrite the following text to remove any toxicity, insults, or profanity "
    "while preserving the original meaning as closely as possible. Only output "
    "the rewritten text.\n\nText: {toxic}"
)


@dataclass(frozen=True)
class GroupedSplitPlan:
    """Indices and audit counts for a normalized-source grouped split."""

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    raw_row_count: int
    unique_source_count: int
    duplicate_row_count: int
    train_source_count: int
    test_source_count: int


def normalize_toxic_source(text: str) -> str:
    """Return the key used to identify equivalent ParaDetox RL prompts.

    NFKC folds compatibility forms such as full-width characters. Splitting and
    joining trims and folds Unicode whitespace, while casefold catches case-only
    duplicates. Punctuation and wording are deliberately preserved.
    """

    if not isinstance(text, str):
        raise TypeError(f"toxic source must be a string, got {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _seeded_source_rank(source_key: str, seed: int) -> bytes:
    """Return a stable pseudo-random rank independent of Python hash state."""

    return hashlib.sha256(f"{seed}\0{source_key}".encode()).digest()


def make_grouped_split_plan(
    toxic_sources: Sequence[str],
    *,
    test_ratio: float = 0.02,
    seed: int = 31415,
    deduplicate_prompts: bool = True,
    row_fingerprints: Sequence[str] | None = None,
) -> GroupedSplitPlan:
    """Plan a deterministic split with no normalized source crossing splits.

    test_ratio is applied to unique normalized sources rather than raw rows.
    row_fingerprints optionally supplies a stable key for selecting one row when
    a source has multiple references; the CLI fingerprints source + reference.
    At least one unique source is assigned to each split.
    """

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be strictly between 0 and 1, got {test_ratio}")

    sources = tuple(toxic_sources)
    if row_fingerprints is None:
        fingerprints = sources
    else:
        fingerprints = tuple(row_fingerprints)
        if len(fingerprints) != len(sources):
            raise ValueError(
                "row_fingerprints must contain one value per toxic source: "
                f"got {len(fingerprints)} fingerprints for {len(sources)} rows"
            )

    groups: dict[str, list[int]] = {}
    for index, source in enumerate(sources):
        source_key = normalize_toxic_source(source)
        if not source_key:
            raise ValueError(f"toxic source at row {index} is empty after normalization")
        groups.setdefault(source_key, []).append(index)

    unique_source_count = len(groups)
    if unique_source_count < 2:
        raise ValueError(
            "at least two unique normalized toxic sources are required for non-empty train/test splits; "
            f"found {unique_source_count}"
        )

    ranked_sources = sorted(groups, key=lambda key: (_seeded_source_rank(key, seed), key))
    test_source_count = min(unique_source_count - 1, max(1, int(unique_source_count * test_ratio)))
    test_sources = ranked_sources[:test_source_count]
    train_sources = ranked_sources[test_source_count:]

    def select_indices(source_keys: Sequence[str]) -> tuple[int, ...]:
        selected: list[int] = []
        for source_key in source_keys:
            candidates = sorted(groups[source_key], key=lambda idx: (fingerprints[idx], sources[idx], idx))
            selected.extend(candidates[:1] if deduplicate_prompts else candidates)
        return tuple(selected)

    train_indices = select_indices(train_sources)
    test_indices = select_indices(test_sources)

    # Keep invariants beside the implementation so later edits fail loudly
    # instead of silently reintroducing train/test leakage.
    train_keys = {normalize_toxic_source(sources[index]) for index in train_indices}
    test_keys = {normalize_toxic_source(sources[index]) for index in test_indices}
    overlap = train_keys & test_keys
    if overlap:
        raise AssertionError(f"normalized toxic-source leakage across splits: {sorted(overlap)[:3]}")
    if set(train_indices) & set(test_indices):
        raise AssertionError("row indices overlap across train and test")
    if len(train_keys) != unique_source_count - test_source_count or len(test_keys) != test_source_count:
        raise AssertionError("split source counts do not match the planned group counts")
    expected_rows = unique_source_count if deduplicate_prompts else len(sources)
    if len(train_indices) + len(test_indices) != expected_rows:
        raise AssertionError(
            f"selected {len(train_indices) + len(test_indices)} rows, expected {expected_rows}"
        )

    return GroupedSplitPlan(
        train_indices=train_indices,
        test_indices=test_indices,
        raw_row_count=len(sources),
        unique_source_count=unique_source_count,
        duplicate_row_count=len(sources) - unique_source_count,
        train_source_count=unique_source_count - test_source_count,
        test_source_count=test_source_count,
    )


def _make_map_fn(split: str, data_source: str):
    def process_fn(example, idx):
        toxic = example["en_toxic_comment"]
        neutral = example["en_neutral_comment"]
        question = DETOX_INSTRUCTION.format(toxic=toxic)

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "alignment",
            "reward_model": {
                "style": "rule",
                "ground_truth": neutral,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question,
                "toxic_comment": toxic,
                "normalized_toxic_source": normalize_toxic_source(toxic),
                "neutral_comment": neutral,
            },
        }

    return process_fn


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local_save_dir", default="./ParaDetox", help="Save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.02,
        help="Fraction of unique normalized sources held out for test (rows in legacy mode).",
    )
    parser.add_argument("--seed", type=int, default=31415, help="Seed for the deterministic train/test split.")
    parser.add_argument(
        "--keep_duplicate_prompts",
        "--keep-duplicate-prompts",
        action="store_true",
        help="Keep every reference row while still assigning source groups atomically.",
    )
    parser.add_argument(
        "--legacy_row_split",
        "--legacy-row-split",
        action="store_true",
        help="Reproduce the old seeded shuffle + row split; prompt leakage is possible.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    # Lazy import keeps the split helpers testable without datasets or network.
    import datasets

    args = _parse_args(argv)
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError(f"test_ratio must be strictly between 0 and 1, got {args.test_ratio}")

    data_source = "s-nlp/paradetox"
    dataset = datasets.load_dataset(data_source, split="train")
    raw_sources = tuple(dataset["en_toxic_comment"])
    normalized_sources = tuple(normalize_toxic_source(source) for source in raw_sources)
    unique_source_count = len(set(normalized_sources))

    if args.legacy_row_split:
        # Preserve the former split semantics: shuffle rows, then slice by rows.
        dataset = dataset.shuffle(seed=args.seed)
        test_num = max(1, int(len(dataset) * args.test_ratio))
        test_dataset = dataset.select(range(test_num))
        train_dataset = dataset.select(range(test_num, len(dataset)))
        train_keys = {normalize_toxic_source(source) for source in train_dataset["en_toxic_comment"]}
        test_keys = {normalize_toxic_source(source) for source in test_dataset["en_toxic_comment"]}

        print("Split mode: legacy row-level shuffle (prompt deduplication disabled)")
        print(f"Raw rows: {len(dataset):,}")
        print(f"Unique normalized toxic sources: {unique_source_count:,}")
        print(f"Duplicate prompt rows: {len(dataset) - unique_source_count:,}")
        print(f"Train rows: {len(train_dataset):,}; test rows: {len(test_dataset):,}")
        print(f"WARNING: normalized sources shared by train/test: {len(train_keys & test_keys):,}")
    else:
        neutral_references = tuple(dataset["en_neutral_comment"])
        row_fingerprints = tuple(
            f"{source}\0{neutral}" for source, neutral in zip(raw_sources, neutral_references, strict=True)
        )
        plan = make_grouped_split_plan(
            raw_sources,
            test_ratio=args.test_ratio,
            seed=args.seed,
            deduplicate_prompts=not args.keep_duplicate_prompts,
            row_fingerprints=row_fingerprints,
        )
        train_dataset = dataset.select(list(plan.train_indices))
        test_dataset = dataset.select(list(plan.test_indices))

        print("Split mode: normalized toxic-source groups")
        print(f"Raw rows: {plan.raw_row_count:,}")
        print(f"Unique normalized toxic sources: {plan.unique_source_count:,}")
        print(f"Duplicate prompt rows in source data: {plan.duplicate_row_count:,}")
        print(f"Prompt deduplication enabled: {not args.keep_duplicate_prompts}")
        print(f"Train: {len(train_dataset):,} rows from {plan.train_source_count:,} source groups")
        print(f"Test: {len(test_dataset):,} rows from {plan.test_source_count:,} source groups")
        print("Normalized sources shared by train/test: 0 (asserted)")

    train_dataset = train_dataset.map(function=_make_map_fn("train", data_source), with_indices=True)
    test_dataset = test_dataset.map(function=_make_map_fn("test", data_source), with_indices=True)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    test_path = os.path.join(args.local_save_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"Saved train dataset to {train_path}")
    print(f"Saved test dataset to {test_path}")


if __name__ == "__main__":
    main()
