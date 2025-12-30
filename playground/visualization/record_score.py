import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

from recipe.amo_safe.reward_client import compute_score

def process_parquet(parquet_path, output_jsonl_path):
    """
    Process parquet file to compute helpful and harmless scores for each response.
    
    Args:
        parquet_path: Path to the input parquet file
        output_jsonl_path: Path to save the output JSONL file
    """
    df = pd.read_parquet(parquet_path)
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded {len(df)} rows from {parquet_path}")
    
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            # Extract extra info containing question metadata
            extra_info = row.get("extra_info", {})
            question = extra_info.get("question", "")
            
            # Get responses and convert numpy array to list if needed
            responses = row.get("responses", [])
            if hasattr(responses, 'tolist'):
                responses = responses.tolist()
            
            # Skip if no valid responses
            if not isinstance(responses, list) or len(responses) == 0:
                continue
            
            # Get the first response and compute scores
            response = responses[0]
            helpful_score, harmless_score = compute_score(question, response)
            
            # Build record with all information
            new_record = {
                **extra_info,
                "response": response,
                "helpful_score": helpful_score,
                "harmless_score": harmless_score,
            }
            
            # Write to JSONL file
            out_f.write(json.dumps(new_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process PKU-SafeRLHF dataset to compute scores")
    parser.add_argument("-e", "--experiment_name", type=str, required=True, help="Name of the experiment to process")
    
    # Parse arguments
    args = parser.parse_args()
    experiment_name = args.experiment_name

    parquet_path = f"results/PKU-SafeRLHF/{experiment_name}.parquet"
    output_jsonl_path = f"playground/visualization/scores/{experiment_name}.jsonl"
    
    process_parquet(parquet_path, output_jsonl_path)
