"""SIM (content preservation) detox reward with the standard Amo signature.

Returns the cosine similarity between the original toxic source sentence and the
model output. The toxic source is read from ``extra_info['toxic_comment']``.
"""

import os

import dotenv
dotenv.load_dotenv()

from recipe.amo_detox.reward_client import compute_reward_score

SIM_TARGET_HOST = os.getenv("SIM_TARGET_HOST", "localhost")
SIM_TARGET_PORT = os.getenv("SIM_TARGET_PORT", "50061")


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    del data_source, ground_truth  # Unused; kept for a uniform Amo metric signature.
    source = extra_info["toxic_comment"] if extra_info and "toxic_comment" in extra_info else ""
    assert source != "", "extra_info['toxic_comment'] must be provided for the SIM axis"
    return compute_reward_score(source, solution_str, SIM_TARGET_HOST, SIM_TARGET_PORT)


if __name__ == "__main__":
    toxic = "you are a complete idiot and everyone hates you"
    extra = {"toxic_comment": toxic}
    print("preserved:", compute_score("detox", "You are wrong and many people dislike your view.", "", extra))
    print("unrelated:", compute_score("detox", "The weather is nice today.", "", extra))
