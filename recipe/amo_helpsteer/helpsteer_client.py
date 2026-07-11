"""Client helpers for the HelpSteer (ArmoRM) multi-attribute reward service."""

import os
from typing import Dict

import grpc

from recipe.amo_helpsteer import reward_pb2, reward_pb2_grpc


ATTRIBUTES = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']


def compute_reward_scores(prompt: str, response: str, host: str, port: str) -> Dict[str, float]:
    """Return the five HelpSteer2 attribute rewards for a (prompt, response) pair."""
    with grpc.insecure_channel(f'{host}:{port}') as channel:
        stub = reward_pb2_grpc.HelpSteerRewardServiceStub(channel)
        request = reward_pb2.ScoreRequest(prompt=prompt, response=response)
        reply = stub.ComputeScore(request)
        return {attr: getattr(reply, attr) for attr in ATTRIBUTES}


def compute_scores(prompt: str, response: str) -> Dict[str, float]:
    """Convenience wrapper reading the target host/port from the environment."""
    host = os.getenv('HELPSTEER_TARGET_HOST', 'localhost')
    port = os.getenv('HELPSTEER_TARGET_PORT', '50054')
    return compute_reward_scores(prompt, response, host, port)


if __name__ == '__main__':
    import time

    import dotenv

    dotenv.load_dotenv()

    prompt = 'What are some synonyms for the word "beautiful"?'
    response = 'Gorgeous, Stunning, Lovely, Elegant, Pretty, Handsome, Wonderful.'

    time_start = time.time()
    scores = compute_scores(prompt, response)
    time_end = time.time()
    print('time_cost:', time_end - time_start)
    for attr, value in scores.items():
        print(f'{attr}: {value:.4f}')
