# Copyright 2026 Rihong Qiu
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
Preprocess the HelpSteer2 dataset to parquet format.

HelpSteer2 (``nvidia/HelpSteer2``) is a multi-attribute helpfulness dataset.
Each sample contains a ``prompt``, a ``response`` and five human-annotated
attribute ratings (each on a 0-4 integer scale):

    - helpfulness
    - correctness
    - coherence
    - complexity
    - verbosity

The processed parquet keeps the prompt (as a chat message list), the raw
response and the five attribute labels as top-level columns so that the data
can be reused for RL rollouts (scored on the fly by the ArmoRM reward service
in ``recipe/amo_helpsteer``), evaluation or supervised analysis.
"""

import argparse
import os

import datasets


# The five HelpSteer2 attributes, kept as top-level label columns.
ATTRIBUTES = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']


def make_map_fn(data_source: str, split: str):
    def process_fn(example, idx):
        question = example.pop('prompt')
        response = example.pop('response')

        # Collect the five human-annotated attribute ratings.
        labels = {attr: example.get(attr, None) for attr in ATTRIBUTES}

        data = {
            'data_source': data_source,
            'prompt': [
                {
                    'role': 'user',
                    'content': question,
                }
            ],
            'response': response,
            'ability': 'alignment',
            'reward_model': {
                # Scored by a reward model service (ArmoRM), not a rule.
                'style': 'model',
                'ground_truth': '',  # should not be used
            },
            'extra_info': {
                'split': split,
                'index': idx,
                'question': question,
                'response': response,
                **labels,
            },
            # Keep the attribute labels as top-level columns for eval / analysis.
            **labels,
        }
        return data

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_save_dir',
        default='./HelpSteer2',
        help='The save directory for the preprocessed dataset.',
    )
    args = parser.parse_args()

    data_source = 'nvidia/HelpSteer2'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    # HelpSteer2 ships a ``validation`` split (no dedicated test split).
    val_dataset = dataset['validation']

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')

    # ``remove_columns`` drops the original flat columns so that only the newly
    # built schema (built inside ``make_map_fn``) is kept.
    train_dataset = train_dataset.map(
        function=make_map_fn(data_source, 'train'),
        with_indices=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        function=make_map_fn(data_source, 'val'),
        with_indices=True,
        remove_columns=val_dataset.column_names,
    )

    local_save_dir = args.local_save_dir
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_save_dir, 'val.parquet'))

    print(f'Saved datasets to {local_save_dir}')


if __name__ == '__main__':
    main()
