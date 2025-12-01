import re

# Boxed answer: at least one non-blank char inside \boxed{…}
BOXED_RE = re.compile(r"\\boxed\{\s*[^}\s][^}]*\}", re.S)

# Generic “enumerator” token (number, letter, roman, or bullet)
ENUMERATOR_RE = re.compile(
    r"""
        (?:                                     # ── Variant A: numbered / lettered / roman
            (?:^|\s|\()                         # left boundary
            (?:\bstep\s*)?                      # optional "step"
            (?:\d+|[a-z]|[ivxlcdm]+)            # token
            [\)\.\:\-]                          # closing punctuation
        )
        |                                       # ── Variant B: bullets
        (?:^|\s)[-*•]\s+                        # boundary + bullet + space
    """,
    re.I | re.X | re.M,                         # ignore-case, verbose, multiline
)

# Inline ordinal words
ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|next|then)\b",
    re.I,
)

def has_step_by_step_thinking(text: str) -> bool:

    enum_lines = ENUMERATOR_RE.findall(text)
    if len(enum_lines) >= 2:
        return True

    ordinals = ORDINAL_RE.findall(text)
    if len(ordinals) >= 2:
        return True

    return False

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:

    if BOXED_RE.search(solution_str) and has_step_by_step_thinking(solution_str):
        return 1
    
    return 0
