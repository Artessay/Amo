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
"""Network-free tests for the ParaDetox grouped split planner."""

import pytest

from data.paradetox import make_grouped_split_plan, normalize_toxic_source


def _keys(sources, indices):
    return {normalize_toxic_source(sources[index]) for index in indices}


def test_normalize_toxic_source_folds_compatibility_case_and_whitespace():
    assert normalize_toxic_source("  ＦＯＯ\tBar\n") == "foo bar"
    assert normalize_toxic_source("Straße") == "strasse"


def test_default_split_deduplicates_and_has_no_prompt_leakage():
    sources = [
        "Bad  sentence",
        " bad sentence ",
        "ＦＯＯ",
        "foo",
        "bar",
        "baz",
        "qux",
    ]
    fingerprints = ["bad-z", "bad-a", "foo-z", "foo-a", "bar", "baz", "qux"]

    first = make_grouped_split_plan(sources, test_ratio=0.4, seed=7, row_fingerprints=fingerprints)
    second = make_grouped_split_plan(sources, test_ratio=0.4, seed=7, row_fingerprints=fingerprints)

    assert first == second
    assert first.raw_row_count == 7
    assert first.unique_source_count == 5
    assert first.duplicate_row_count == 2
    assert first.test_source_count == 2
    assert len(first.train_indices) + len(first.test_indices) == 5
    assert _keys(sources, first.train_indices).isdisjoint(_keys(sources, first.test_indices))

    # The lowest stable fingerprint wins rather than a post-shuffle occurrence.
    selected = set(first.train_indices + first.test_indices)
    assert 1 in selected and 0 not in selected
    assert 3 in selected and 2 not in selected


def test_keep_duplicates_retains_all_rows_without_cross_split_overlap():
    sources = ["same source", " SAME\tSOURCE ", "one", "two", "three"]
    plan = make_grouped_split_plan(
        sources,
        test_ratio=0.5,
        seed=11,
        deduplicate_prompts=False,
    )

    assert len(plan.train_indices) + len(plan.test_indices) == len(sources)
    assert set(plan.train_indices + plan.test_indices) == set(range(len(sources)))
    assert _keys(sources, plan.train_indices).isdisjoint(_keys(sources, plan.test_indices))


def test_test_size_is_computed_from_unique_sources_not_raw_rows():
    sources = (["repeated"] * 20) + ["one", "two", "three"]
    plan = make_grouped_split_plan(sources, test_ratio=0.5, seed=3)

    assert plan.unique_source_count == 4
    assert plan.test_source_count == 2
    assert plan.train_source_count == 2
    assert len(plan.test_indices) == 2
    assert len(plan.train_indices) == 2


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_invalid_test_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        make_grouped_split_plan(["one", "two"], test_ratio=ratio)


def test_empty_and_single_source_inputs_are_rejected():
    with pytest.raises(ValueError, match="empty after normalization"):
        make_grouped_split_plan([" \t", "valid"], test_ratio=0.5)
    with pytest.raises(ValueError, match="at least two unique"):
        make_grouped_split_plan(["same", " SAME "], test_ratio=0.5)


def test_fingerprint_length_must_match_rows():
    with pytest.raises(ValueError, match="one value per toxic source"):
        make_grouped_split_plan(["one", "two"], test_ratio=0.5, row_fingerprints=["only-one"])
