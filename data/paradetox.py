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
"""
Preprocess the ParaDetox dataset to parquet format.

ParaDetox (s-nlp/paradetox) is a parallel text detoxification corpus. Each row
pairs a toxic sentence with a neutral rewrite that removes toxicity while
preserving meaning. It ships with a single ``train`` split, so we carve out a
small held-out test split here.

The task is framed as multi-objective alignment with three conflicting axes
scored by the reward servers under ``recipe/amo_detox``:
  * STA  - style transfer accuracy (toxicity removed)
  * SIM  - content preservation (semantic similarity to the *toxic* source)
  * FL   - fluency (linguistic acceptability)

Because the SIM axis needs the original toxic sentence at scoring time, we stash
it in ``extra_info['toxic_comment']``.
"""

import argparse
import os

import datasets


DETOX_INSTRUCTION = (
    "Rewrite the following text to remove any toxicity, insults, or profanity "
    "while preserving the original meaning as closely as possible. Only output "
    "the rewritten text.\n\nText: {toxic}"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir", default="./ParaDetox", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.02, help="Fraction of data held out for the test split."
    )
    parser.add_argument("--seed", type=int, default=31415, help="Shuffle seed for the train/test split.")
    args = parser.parse_args()

    data_source = "s-nlp/paradetox"

    # ParaDetox only provides a single 'train' split.
    dataset = datasets.load_dataset(data_source, split="train")
    dataset = dataset.shuffle(seed=args.seed)

    test_num = max(1, int(len(dataset) * args.test_ratio))
    test_dataset = dataset.select(range(test_num))
    train_dataset = dataset.select(range(test_num, len(dataset)))

    print(f"Total dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    def make_map_fn(split):
        def process_fn(example, idx):
            toxic = example["en_toxic_comment"]
            neutral = example["en_neutral_comment"]

            question = DETOX_INSTRUCTION.format(toxic=toxic)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "alignment",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": neutral,  # reference neutral rewrite
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    # SIM axis compares the model output against the toxic source.
                    "toxic_comment": toxic,
                    "neutral_comment": neutral,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_save_dir
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Saved datasets to {local_save_dir}")
