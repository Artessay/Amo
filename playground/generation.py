import os
import torch
import random
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def enable_full_determinism(seed: int):
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def load_model_and_tokenizer(model_path: str, gpu_memory_utilization: float = 0.9):
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # initialize the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=gpu_memory_utilization)
    
    return llm, tokenizer

def inference(llm: LLM, prompts: list, sampling_params: SamplingParams):
    # generate
    outputs = llm.generate(prompts, sampling_params)

    # convert to text
    output_list = []
    token_counts = []
    for output in outputs:
        responses = [resp.text.strip() for resp in output.outputs]
        output_list.append(responses)
        # Get token counts
        token_count = sum(len(resp.token_ids) for resp in output.outputs)
        token_counts.append(token_count)
    return output_list, token_counts


def generate(args):
    seed = args.seed
    data_path = args.data
    model_path = args.model
    output_path = args.output
    max_tokens = args.max_tokens
    gpu_memory_utilization = args.gpu_memory_utilization

    enable_full_determinism(seed)

    llm, tokenizer = load_model_and_tokenizer(model_path, gpu_memory_utilization)
    print(f"Start inference on {data_path} with {model_path}")

    # read dataset.
    dataframe = pd.read_parquet(data_path)
    prompts = [
        tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False,
        ) 
        for prompt in dataframe["prompt"]
    ]
    
    sampling_params = SamplingParams(n=1, max_tokens=max_tokens, seed=seed)
    output_list, token_counts = inference(llm, prompts, sampling_params)

    # add to the data frame
    dataframe["responses"] = output_list
    
    # Add num_tokens to extra_info
    assert "extra_info" in dataframe.columns, "extra_info column is required"
    
    # Create a new list of updated extra_info dictionaries
    updated_extra_info = []
    for i, row in dataframe.iterrows():
        extra_info = row.get("extra_info", {})
        extra_info["num_tokens"] = token_counts[i]
        updated_extra_info.append(extra_info)
    
    # Assign the entire list to the column at once
    dataframe["extra_info"] = updated_extra_info

    # write to a new parquet
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    dataframe.to_parquet(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Amo Benchmark")

    # pretrain, finetune, continue train
    parser.add_argument('-d', '--data', type=str, help='data path')
    
    parser.add_argument('-m', '--model', type=str, help='model path')

    parser.add_argument('-o', '--output', type=str, help='output path')

    parser.add_argument('-t', '--max_tokens', type=int, default=1024, help='max tokens')

    parser.add_argument('-g', '--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')

    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    print(args)

    generate(args)