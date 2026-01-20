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
                "article": article,
                "ground_truth": highlights,
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
    parser.add_argument(
        "--train_percentage",
        type=float,
        default=5.0,
        help="Percentage of training data to use (0.0 to 100.0). Default: 1.0",
    )
    parser.add_argument(
        "--test_percentage",
        type=float,
        default=5.0,
        help="Percentage of test data to use (0.0 to 100.0). Default: 5.0",
    )

    args = parser.parse_args()

    data_source = "abisee/cnn_dailymail"

    dataset = datasets.load_dataset(data_source, "3.0.0")

    # Calculate sample size (1% of the original data)
    train_fraction = args.train_percentage / 100.0
    test_fraction = args.test_percentage / 100.0
    
    # Sample each split
    train_dataset = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * train_fraction)))
    val_dataset = dataset["validation"].shuffle(seed=42).select(range(int(len(dataset["validation"]) * test_fraction)))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(int(len(dataset["test"]) * test_fraction)))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

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