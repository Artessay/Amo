"""gRPC server that wraps the ArmoRM multi-attribute reward model.

The server loads ``RLHFlow/ArmoRM-Llama3-8B-v0.1`` once and keeps it resident
in memory. For each request it scores a (prompt, response) pair and returns the
five HelpSteer2 attribute rewards produced by ArmoRM:

    helpfulness, correctness, coherence, complexity, verbosity

ArmoRM outputs a 19-dimensional multi-objective reward vector; the first five
entries correspond to the HelpSteer attributes (in the order above). We return
the raw ArmoRM reward values without rescaling.
"""

import logging
from concurrent import futures
from typing import Dict

import grpc
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import reward_pb2
import reward_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
MAX_LENGTH = 4096

# The five HelpSteer2 attributes are the first five ArmoRM objectives.
HELPSTEER_ATTRIBUTES = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
HELPSTEER_SLICE = slice(0, 5)


class HelpSteerRewardServiceServicer(reward_pb2_grpc.HelpSteerRewardServiceServicer):
    """gRPC servicer wrapping ArmoRM for multi-attribute reward scoring."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info('Loading ArmoRM reward model from %s onto %s ...', model_path, self.device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        logger.info('ArmoRM reward model loaded successfully.')

    def ComputeScore(self, request, context):  # pylint: disable=unused-argument
        """Compute the five HelpSteer2 attribute rewards for one pair."""
        scores = self.compute_helpsteer_scores(request.prompt, request.response)
        return reward_pb2.ScoreResponse(**scores)

    @torch.no_grad()
    def compute_helpsteer_scores(self, prompt: str, response: str) -> Dict[str, float]:
        """Return a dict of the five HelpSteer2 attribute rewards."""
        messages = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.model.device)

        output = self.model(input_ids)
        # output.rewards: size = (B, 19); take the first sample, first 5 objectives.
        rewards = output.rewards.cpu().float()[0, HELPSTEER_SLICE].tolist()

        scores = {attr: float(value) for attr, value in zip(HELPSTEER_ATTRIBUTES, rewards)}
        logger.info(
            '\nPrompt: %s\nResponse: %s\nScores: %s',
            prompt,
            response,
            ', '.join(f'{k}={v:.4f}' for k, v in scores.items()),
        )
        return scores


def serve(model_path: str, port: int) -> None:
    """Start the HelpSteer reward gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    reward_pb2_grpc.add_HelpSteerRewardServiceServicer_to_server(
        HelpSteerRewardServiceServicer(model_path),
        server,
    )
    server.add_insecure_port(f'[::]:{port}')
    logger.info('Starting HelpSteerRewardService on port %d', port)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Start the HelpSteer (ArmoRM) gRPC server.')
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='Path or HF id of the ArmoRM reward model.',
    )
    parser.add_argument('--port', type=int, default=50054, help='Port to run the server on.')
    args = parser.parse_args()

    serve(model_path=args.model_path, port=args.port)
