"""gRPC server for the STA (style transfer accuracy) detox reward.

Wraps ``s-nlp/roberta_toxicity_classifier`` and returns the probability that the
model output is *non-toxic* (i.e. the "neutral" class), following the standard
ParaDetox evaluation protocol.
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


class StaServiceServicer(reward_pb2_grpc.RewardServiceServicer):
    """Return P(non-toxic) for the model output."""

    def __init__(self, model_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def ComputeScore(self, request, context):  # pylint: disable=unused-argument
        score = self.compute_sta_score(request.response)
        return reward_pb2.ScoreResponse(reward_score=score)

    def compute_sta_score(self, response: str) -> float:
        inputs = self.tokenizer(
            response,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).view(-1)

        # s-nlp/roberta_toxicity_classifier: index 0 = neutral, index 1 = toxic.
        neutral_prob = float(probs[0].item())
        logger.info("\nResponse: %s\nSTA (neutral prob): %.6f", response, neutral_prob)
        return neutral_prob


def serve(model_path: str, port: int) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    reward_pb2_grpc.add_RewardServiceServicer_to_server(StaServiceServicer(model_path), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting STA RewardService on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the STA detox reward gRPC server.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the toxicity classifier.")
    parser.add_argument("--port", type=int, default=50060, help="Port to run the server on.")
    args = parser.parse_args()

    serve(model_path=args.model_path, port=args.port)
