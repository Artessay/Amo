import os
import torch
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from verl.workers.engine.utils import enable_full_determinism

def load_model_and_tokenizer(model_path: str):
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # initialize the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.9)
    
    return llm, tokenizer

def inference(llm: LLM, prompts: list, sampling_params: SamplingParams):
    # generate
    outputs = llm.generate(prompts, sampling_params)

    # convert to text
    output_list = []
    for output in outputs:
        responses = [resp.text.strip() for resp in output.outputs]
        output_list.append(responses)
    return output_list


def generate(args):
    seed = args.seed
    data_path = args.data
    model_path = args.model
    output_path = args.output
    max_tokens = args.max_tokens

    enable_full_determinism(seed)

    llm, tokenizer = load_model_and_tokenizer(model_path)
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
    output_list = inference(llm, prompts, sampling_params)

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
    parser.add_argument('-d', '--data', type=str, help='data path')
    
    parser.add_argument('-m', '--model', type=str, help='model path')

    parser.add_argument('-o', '--output', type=str, help='output path')

    parser.add_argument('-t', '--max_tokens', type=int, default=2048, help='max tokens')

    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    print(args)

    generate(args)