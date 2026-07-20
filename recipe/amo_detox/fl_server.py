"""gRPC server for the FL (fluency) detox reward.

Wraps a RoBERTa-based CoLA (Corpus of Linguistic Acceptability) classifier and
returns the probability that the model output is linguistically *acceptable*,
following the standard ParaDetox fluency protocol.
"""

import argparse
import logging
from concurrent import futures

import grpc
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import reward_pb2
import reward_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 512


class FlServiceServicer(reward_pb2_grpc.RewardServiceServicer):
    """Return P(acceptable) for the model output."""

    def __init__(self, model_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def ComputeScore(self, request, context):  # pylint: disable=unused-argument
        score = self.compute_fl_score(request.response)
        return reward_pb2.ScoreResponse(reward_score=score)

    def compute_fl_score(self, response: str) -> float:
        inputs = self.tokenizer(
            response,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).view(-1)

        # textattack/roberta-base-CoLA: index 0 = unacceptable, index 1 = acceptable.
        acceptable_prob = float(probs[1].item())
        logger.info("\nResponse: %s\nFL (acceptable prob): %.6f", response, acceptable_prob)
        return acceptable_prob


def serve(model_path: str, port: int) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    reward_pb2_grpc.add_RewardServiceServicer_to_server(FlServiceServicer(model_path), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting FL RewardService on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the FL detox reward gRPC server.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="textattack/roberta-base-CoLA",
        help="Path or name of the CoLA acceptability classifier.",
    )
    parser.add_argument("--port", type=int, default=50062, help="Port to run the server on.")
    args = parser.parse_args()

    serve(model_path=args.model_path, port=args.port)
