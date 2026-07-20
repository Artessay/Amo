"""STA (style transfer accuracy) detox reward with the standard Amo signature.

Returns the probability that the model output is non-toxic.
"""

import os

import dotenv
dotenv.load_dotenv()

from recipe.amo_detox.reward_client import compute_reward_score

STA_TARGET_HOST = os.getenv("STA_TARGET_HOST", "localhost")
STA_TARGET_PORT = os.getenv("STA_TARGET_PORT", "50060")


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    del data_source, ground_truth  # Unused; kept for a uniform Amo metric signature.
    # STA only needs the model output.
    return compute_reward_score("", solution_str, STA_TARGET_HOST, STA_TARGET_PORT)


if __name__ == "__main__":
    toxic = "you are a complete idiot and everyone hates you"
    print("clean :", compute_score("detox", "I disagree with you and I think many people do too.", ""))
    print("toxic :", compute_score("detox", toxic, ""))
