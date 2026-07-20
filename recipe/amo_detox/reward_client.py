"""Shared gRPC client helper for the detox reward servers."""

import grpc

from recipe.amo_detox import reward_pb2, reward_pb2_grpc


def compute_reward_score(prompt: str, response: str, host: str, port: str) -> float:
    """Call a detox reward gRPC server and return its scalar score.

    Args:
        prompt: For SIM this is the original toxic source sentence; for STA/FL
            it is unused (pass an empty string).
        response: The model-generated (detoxified) output.
        host: Server host.
        port: Server port.
    """
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = reward_pb2_grpc.RewardServiceStub(channel)
        request = reward_pb2.ScoreRequest(prompt=prompt, response=response)
        reply = stub.ComputeScore(request)
        return reply.reward_score
