"""Launch one ParaDetox reward server from a local HF snapshot.

This wrapper keeps the original reward implementations unchanged while making
repo-id loading reliable in offline environments. Some transformers versions
try a Hub metadata request when LaBSE is passed as a repo id; resolving the id
to its cached snapshot first avoids that request.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_IDS = {
    "sta": "s-nlp/roberta_toxicity_classifier",
    "sim": "sentence-transformers/LaBSE",
    "fl": "textattack/roberta-base-CoLA",
}
DEFAULT_PORTS = {"sta": 50060, "sim": 50061, "fl": 50062}


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path.resolve())

    offline = os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
    try:
        return snapshot_download(repo_id=model_path, local_files_only=offline)
    except Exception as exc:
        mode = "offline cache" if offline else "Hugging Face Hub"
        raise RuntimeError(f"Could not resolve {model_path!r} from the {mode}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("axis", choices=MODEL_IDS)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path or MODEL_IDS[args.axis])
    port = DEFAULT_PORTS[args.axis] if args.port is None else args.port
    if not 1 <= port <= 65535:
        parser.error("--port must be between 1 and 65535")
    print(f"Starting {args.axis.upper()} reward server on :{port} from {model_path}", flush=True)

    if args.axis == "sta":
        from sta_server import serve
    elif args.axis == "sim":
        from sim_server import serve
    else:
        from fl_server import serve
    # The imported legacy modules call basicConfig(level=INFO). Reset logging
    # afterwards so the default quiet mode does not log every reward request.
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.WARNING)
    serve(model_path=model_path, port=port)


if __name__ == "__main__":
    main()
