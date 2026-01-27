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
Preprocess the PKU-SafeRLHF dataset to parquet format
"""

import argparse
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir", default="./PKU-SafeRLHF", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--train_percentage",
        type=float,
        default=5.0,
        help="Percentage of training data to use (0.0 to 100.0). Default: 1.0",
    )
    parser.add_argument(
        "--test_percentage",
        type=float,
        default=10.0,
        help="Percentage of test data to use (0.0 to 100.0). Default: 5.0",
    )

    args = parser.parse_args()

    data_source = "PKU-Alignment/PKU-SafeRLHF"
    dataset = datasets.load_dataset(data_source)

    # Calculate sample size (1% of the original data)
    train_fraction = args.train_percentage / 100.0
    test_fraction = args.test_percentage / 100.0
    
    # Sample each split
    train_dataset = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * train_fraction)))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(int(len(dataset["test"]) * test_fraction)))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")
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
                    "ground_truth": "",  # should not be used
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
