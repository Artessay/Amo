"""gRPC server for the SIM (content preservation) detox reward.

Uses the LaBSE sentence encoder to compute the cosine similarity between the
original toxic source sentence (``request.prompt``) and the model output
(``request.response``). Higher similarity means more meaning was preserved.
"""

import argparse
import logging
from concurrent import futures

import grpc
import torch
from sentence_transformers import SentenceTransformer

import reward_pb2
import reward_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimServiceServicer(reward_pb2_grpc.RewardServiceServicer):
    """Return cosine similarity between the toxic source and the model output."""

    def __init__(self, model_path: str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)

    def ComputeScore(self, request, context):  # pylint: disable=unused-argument
        score = self.compute_sim_score(request.prompt, request.response)
        return reward_pb2.ScoreResponse(reward_score=score)

    def compute_sim_score(self, source: str, response: str) -> float:
        if not source or not response:
            return 0.0

        embeddings = self.model.encode(
            [source, response],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        # normalized embeddings -> dot product is cosine similarity
        similarity = float(torch.dot(embeddings[0], embeddings[1]).item())
        logger.info(
            "\nSource: %s\nResponse: %s\nSIM (cosine): %.6f", source, response, similarity
        )
        return similarity


def serve(model_path: str, port: int) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    reward_pb2_grpc.add_RewardServiceServicer_to_server(SimServiceServicer(model_path), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting SIM RewardService on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the SIM detox reward gRPC server.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="sentence-transformers/LaBSE",
        help="Path or name of the LaBSE sentence encoder.",
    )
    parser.add_argument("--port", type=int, default=50061, help="Port to run the server on.")
    args = parser.parse_args()

    serve(model_path=args.model_path, port=args.port)
