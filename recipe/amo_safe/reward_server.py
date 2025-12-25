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
from align_anything.utils.reward_calibration import calibrate_score, load_calibration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 1024


class RewardServiceServicer(reward_pb2_grpc.RewardServiceServicer):
    """gRPC servicer that wraps a reward model for scoring.

    The service keeps the reward model and tokenizer in memory to avoid
    reloading them for every request.
    """

    def __init__(self, model_path: str, calibration_path: Optional[str] = None) -> None:
        """Initialize the reward model, tokenizer, and optional calibration.

        Args:
            model_path: Path to the reward model checkpoint.
            calibration_path: Optional path to a JSON file with calibration
                parameters {"alpha": float, "beta": float}. If provided and
                valid, the server will return calibrated scores in [0, 1].
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

        self._calibration: Optional[Tuple[float, float]] = None
        if calibration_path is not None:
            if os.path.isfile(calibration_path):
                alpha, beta = load_calibration(calibration_path)
                self._calibration = (alpha, beta)
                logger.info(
                    "Loaded reward calibration from %s (alpha=%.6f, beta=%.6f)",
                    calibration_path,
                    alpha,
                    beta,
                )
            else:
                logger.warning(
                    "Calibration file not found at %s; continuing without calibration.",
                    calibration_path,
                )

    def ComputeScore(self, request, context):  # pylint: disable=unused-argument
        """Compute a reward score for a single prompt/response pair."""

        score = self.compute_helpful_score_local(request.prompt, request.response)
        return reward_pb2.ScoreResponse(reward_score=score)

    def compute_helpful_score_local(self, prompt: str, response: str) -> float:
        """Compute reward score for a prompt-response pair.

        If calibration is active, this returns the calibrated score in [0, 1];
        otherwise it returns the raw logit. Both raw and calibrated scores are
        logged when calibration is enabled.
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
            raw_score = float(score_tensor.view(-1)[0].item())

        if self._calibration is not None:
            alpha, beta = self._calibration
            calibrated_tensor = calibrate_score(score_tensor.view(-1), alpha=alpha, beta=beta)
            calibrated_score = float(calibrated_tensor[0].item())
            logger.debug(
                "\nPrompt: %s\nResponse: %s\nRaw score: %.4f\nCalibrated score: %.4f",
                prompt,
                response,
                raw_score,
                calibrated_score,
            )
            return calibrated_score

        logger.debug("\nPrompt: %s\nResponse: %s\nScore: %.6f", prompt, response, raw_score)
        return raw_score


def serve(model_path: str, port: int, calibration_path: Optional[str] = None) -> None:
    """Start the reward gRPC server."""

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    reward_pb2_grpc.add_RewardServiceServicer_to_server(
        RewardServiceServicer(model_path, calibration_path=calibration_path),
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
    parser.add_argument(
        "--calibration_path",
        type=str,
        default=None,
        help="Optional path to a JSON calibration file (alpha/beta).",
    )
    args = parser.parse_args()

    serve(model_path=args.model_path, port=args.port, calibration_path=args.calibration_path)
