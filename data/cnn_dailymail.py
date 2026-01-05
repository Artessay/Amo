"""
Preprocess the CNN/DailyMail dataset to parquet format
"""

import argparse
import os

import datasets


INSTRUCTION = "Summarize the following article in 2-3 sentences."


def make_map_fn(data_source: str, split: str):
    def process_fn(example, idx):
        # Pop original fields so that they are not kept as top-level columns
        article = example.pop("article")
        highlights = example.pop("highlights")
        example_id = example.pop("id", None)
        example_index = example.pop("index", idx)

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": article,
                }
            ],
            "ability": "summarization",
            "reward_model": {
                "style": "rule",
                "ground_truth": highlights,
            },
            "extra_info": {
                "split": split,
                "index": example_index,
                "id": example_id,
                # "article": article,
                # "highlights": highlights,
            },
        }
        return data

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="./cnn_dailymail",
        help="Directory to save the preprocessed parquet files.",
    )

    args = parser.parse_args()

    data_source = "abisee/cnn_dailymail"

    dataset = datasets.load_dataset(data_source, "3.0.0")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    def process_and_save(split_dataset: datasets.Dataset, split_name: str, filename: str):
        mapped = split_dataset.map(
            function=make_map_fn(data_source, split_name),
            with_indices=True,
        )
        mapped.to_parquet(os.path.join(args.local_dir, filename))

    os.makedirs(args.local_dir, exist_ok=True)

    process_and_save(train_dataset, "train", "train.parquet")
    process_and_save(val_dataset, "validation", "val.parquet")
    process_and_save(test_dataset, "test", "test.parquet")


if __name__ == "__main__":
    main()
