"""HelpSteer2 `verbosity` reward metric (ArmoRM-backed).

Exposes the standard Amo `compute_score` signature used by the reward manager
and offline evaluator.
"""

import dotenv

dotenv.load_dotenv()

from recipe.amo_helpsteer.helpsteer_client import compute_scores

DIMENSION = 'verbosity'


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """Return the ArmoRM `verbosity` reward for a single response.

    Args:
        data_source: Dataset name. Unused but kept for a uniform signature.
        solution_str: Model-generated response to score.
        ground_truth: Reference answer. Unused for model-based scoring.
        extra_info: Per-example metadata; must contain the `question` (prompt).

    Returns:
        The `verbosity` attribute reward.
    """
    del data_source, ground_truth  # Unused, kept for a uniform Amo signature.
    question = extra_info['question'] if extra_info and 'question' in extra_info else ''
    assert question != '', 'question must be provided'

    return compute_scores(question, solution_str)[DIMENSION]


if __name__ == '__main__':
    prompt = 'What are some synonyms for the word "beautiful"?'
    response = 'Gorgeous, Stunning, Lovely, Elegant, Pretty, Handsome, Wonderful.'
    print(compute_score('helpsteer2', response, '', extra_info={'question': prompt}))
