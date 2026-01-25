import os
import pandas as pd

data_dir = './data'

# Iterate through all subdirectories in the data directory
for dataset in os.listdir(data_dir):
    dataset_path = os.path.join(data_dir, dataset)
    
    # Only process directories
    if not os.path.isdir(dataset_path):
        continue
    
    print(f"\nDataset: {dataset}")
    
    # Count train.parquet
    train_path = os.path.join(dataset_path, 'train.parquet')
    if os.path.exists(train_path):
        df_train = pd.read_parquet(train_path)
        print(f"  train.parquet: {len(df_train):,} rows")
    else:
        print("  train.parquet: not exists")
    
    # Count test.parquet
    test_path = os.path.join(dataset_path, 'test.parquet')
    if os.path.exists(test_path):
        df_test = pd.read_parquet(test_path)
        print(f"  test.parquet: {len(df_test):,} rows")
    else:
        print("  test.parquet: not exists")
