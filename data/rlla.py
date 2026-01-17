# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the RLLA dataset to parquet format
"""

import os
import json
import numpy as np
import pandas as pd
import argparse

np.random.seed(31415)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./rlla_4k')
    args = parser.parse_args()
    
    data_source = 'rlla'

    # Load dataset
    dataset = json.load(open("./dataset/rlla_4k_raw/rlla_rl.json", "r"))

    # Shuffle dataset
    np.random.shuffle(dataset)

    # Split into train and test sets (2% test data)
    test_num = int(len(dataset) * 0.02)
    train_dataset = dataset[:-test_num]
    test_dataset = dataset[-test_num:]

    # Function to process each example
    def process_fn(example, idx, split):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]

        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": output
            },
            "extra_info": {
                'split': split,
                'index': idx,
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
        }
        return data

    # Process dataset using list comprehension
    train_dataset = [process_fn(d, idx, 'train') for idx, d in enumerate(train_dataset)]
    test_dataset = [process_fn(d, idx, 'test') for idx, d in enumerate(test_dataset)]

    # Convert to Pandas DataFrame
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    # Save as Parquet
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved datasets to {local_dir}")