import os
import torch
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def load_model_and_tokenizer(model_path: str):
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # initialize the model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.9)
    
    return llm, tokenizer

def inference(llm: LLM, prompts: list):
    # generate
    sampling_params = SamplingParams(n=1, max_tokens=4096, seed=42)
    outputs = llm.generate(prompts, sampling_params)

    # convert to text
    output_list = []
    for output in outputs:
        responses = [resp.text.strip() for resp in output.outputs]
        output_list.append(responses)
    return output_list


def generate(args):
    data_path = args.data
    model_path = args.model
    output_path = args.output

    llm, tokenizer = load_model_and_tokenizer(model_path)

    # read dataset.
    dataframe = pd.read_parquet(data_path)
    prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) 
        for prompt in dataframe["prompt"]
    ]
    
    output_list = inference(llm, prompts)

    # add to the data frame
    dataframe["responses"] = output_list

    # write to a new parquet
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    dataframe.to_parquet(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Amo Benchmark")

    # pretrain, finetune, continue train
    parser.add_argument('-d', '--data', type=str, default='data/MATH-500/test.parquet', help='data path')
    
    parser.add_argument('-m', '--model', type=str, default='/data/Qwen/Qwen3-4B', help='model path')

    parser.add_argument('-o', '--output', type=str, default='results/MATH-500/qwen3-4b.parquet', help='output path')
    
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    print(args)

    generate(args)