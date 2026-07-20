"""FL (fluency) detox reward with the standard Amo signature.

Returns the probability that the model output is linguistically acceptable.
"""

import os

import dotenv
dotenv.load_dotenv()

from recipe.amo_detox.reward_client import compute_reward_score

FL_TARGET_HOST = os.getenv("FL_TARGET_HOST", "localhost")
FL_TARGET_PORT = os.getenv("FL_TARGET_PORT", "50062")


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    del data_source, ground_truth  # Unused; kept for a uniform Amo metric signature.
    # FL only needs the model output.
    return compute_reward_score("", solution_str, FL_TARGET_HOST, FL_TARGET_PORT)


if __name__ == "__main__":
    print("fluent :", compute_score("detox", "I respectfully disagree with your opinion.", ""))
    print("broken :", compute_score("detox", "disagree i your the opinion respect not with", ""))
