import os
import logging
from concurrent import futures
from typing import Optional, Tuple

import grpc
import torch

import reward_pb2
import reward_pb2_grpc
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.configs.format_model import ModelFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 1024


class RewardServiceServicer(reward_pb2_grpc.RewardServiceServicer):
    """gRPC servicer that wraps a reward model for scoring.

    The service keeps the reward model and tokenizer in memory to avoid
    reloading them for every request.
    """

    def __init__(self, model_path: str) -> None:
        """Initialize the reward model, tokenizer.

        Args:
            model_path: Path to the reward model checkpoint.
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

    def ComputeScore(self, request, context):  # pylint: disable=unused-argument
        """Compute a reward score for a single prompt/response pair."""

        score = self.compute_helpful_score_local(request.prompt, request.response)
        return reward_pb2.ScoreResponse(reward_score=score)

    def compute_helpful_score_local(self, prompt: str, response: str) -> float:
        """Compute reward score for a prompt-response pair.

        Returns the sigmoid of the logit score.
        """

        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        formatted = self.formatter(conversation)

        inputs = self.tokenizer(
            formatted,
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model(**inputs)
            score_tensor = output.end_scores
            raw_score = score_tensor.view(-1)[0]
            sigmoid_score = torch.sigmoid(raw_score).item()

        logger.info("\nPrompt: %s\nResponse: %s\nScore: %.6f", prompt, response, sigmoid_score)
        return sigmoid_score


def serve(model_path: str, port: int) -> None:
    """Start the reward gRPC server."""

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    reward_pb2_grpc.add_RewardServiceServicer_to_server(
        RewardServiceServicer(model_path),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting RewardService on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the Reward gRPC server.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the reward model.")
    parser.add_argument("--port", type=int, default=50051, help="Port to run the server on.")
    args = parser.parse_args()

    serve(model_path=args.model_path, port=args.port)
