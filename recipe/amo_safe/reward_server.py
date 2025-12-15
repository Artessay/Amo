import torch
from concurrent import futures
import grpc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import reward_pb2, reward_pb2_grpc
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.configs.format_model import ModelFormatter

MAX_PROMPT_LENGTH = 1024

class RewardServiceServicer(reward_pb2_grpc.RewardServiceServicer):
    """
    RewardServiceServicer holds the reward model and tokenizer in memory
    to avoid reloading the model for every evaluation.
    """
    def __init__(self, model_path: str):
        """
        Initialize the reward model and tokenizer. Load only once.
        """
        self.model_path = model_path
        self.model, self.tokenizer, _ = load_pretrained_models(
            model_path,
            model_max_length=MAX_PROMPT_LENGTH,
            padding_side="right",
            is_reward_model=True,
            auto_device_mapping=True,
        )
        self.model = self.model.eval()
        self.formatter = ModelFormatter(self.tokenizer)

    def ComputeScore(self, request, context):
        score = self.compute_helpful_score_local(request.prompt, request.response)
        return reward_pb2.ScoreResponse(reward_score=score)

    
    def compute_helpful_score_local(self, prompt: str, response: str) -> float:
        """
        Compute reward score for a prompt-response pair.
        """
        conversation = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response},
        ]
        formatted = self.formatter(conversation)

        inputs = self.tokenizer(
            formatted,
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model(**inputs)
            score = output.end_scores.item()
        
        logger.info(f'\nPrompt: {prompt}\nResponse: {response}\nScore: {score}')

        return score

def serve(model_path: str, port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    reward_pb2_grpc.add_RewardServiceServicer_to_server(RewardServiceServicer(model_path), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Start the Reward gRPC server.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the reward model.')
    parser.add_argument('--port', type=int, default=50051, help='Port to run the server on.')
    args = parser.parse_args()

    serve(model_path=args.model_path, port=args.port)