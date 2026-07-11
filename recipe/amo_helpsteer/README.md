# HelpSteer2 Multi-Attribute Reward Service (ArmoRM)

This recipe deploys [`RLHFlow/ArmoRM-Llama3-8B-v0.1`](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)
as a gRPC reward service that scores a `(prompt, response)` pair along the five
[HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) attributes:

- `helpfulness`
- `correctness`
- `coherence`
- `complexity`
- `verbosity`

ArmoRM produces a 19-dimensional multi-objective reward vector; the first five
objectives are exactly these HelpSteer attributes. The service returns the raw
ArmoRM reward values (no rescaling). It follows the same deployment pattern as
`recipe/amo_safe` (single reward server) and `recipe/amo_news` (multi-dimension
evaluation).

## 1. Prepare the dataset

Process the HelpSteer2 dataset to parquet (mirrors the other `data/` scripts):

```bash
cd data
python helpsteer2.py --local_save_dir ./HelpSteer2
```

This writes `data/HelpSteer2/train.parquet` and `data/HelpSteer2/val.parquet`.
Each row keeps the `prompt` (chat message list), the raw `response`, and the
five human-annotated attribute labels (0-4) as top-level columns plus in
`extra_info`. See `data/HelpSteer2/example.json` for the schema.

## 2. Compile the protobufs (only if you edit `reward.proto`)

```bash
# pip install grpcio-tools
bash recipe/amo_helpsteer/compile.sh
```

This regenerates `reward_pb2.py` and `reward_pb2_grpc.py` (already checked in).

## 3. Start the reward server

```bash
# Uses RLHFlow/ArmoRM-Llama3-8B-v0.1 by default (downloaded from HF).
# Override REWARD_MODEL_PATH to use a local checkpoint.
bash recipe/amo_helpsteer/start_server.sh
```

The server loads the model once (bf16, on GPU if available) and listens on
port `50054`. A GPU with ~16GB+ of memory is recommended.

## 4. Query the service

```bash
python recipe/amo_helpsteer/helpsteer_client.py
```

Or from Python:

```python
from recipe.amo_helpsteer.helpsteer_client import compute_scores

scores = compute_scores(
    prompt='What are some synonyms for the word "beautiful"?',
    response='Gorgeous, Stunning, Lovely, Elegant, Pretty.',
)
# {'helpfulness': ..., 'correctness': ..., 'coherence': ...,
#  'complexity': ..., 'verbosity': ...}
```

The target host/port can be set via `HELPSTEER_TARGET_HOST` /
`HELPSTEER_TARGET_PORT` (defaults `localhost:50054`).

## 5. Per-dimension metric wrappers

Each attribute has a thin wrapper exposing the standard Amo `compute_score`
signature (`data_source, solution_str, ground_truth, extra_info`), so it can be
plugged into the reward manager / offline evaluator just like `amo_safe` and
`amo_news`:

- `helpsteer_helpfulness.py`
- `helpsteer_correctness.py`
- `helpsteer_coherence.py`
- `helpsteer_complexity.py`
- `helpsteer_verbosity.py`

```python
from recipe.amo_helpsteer.helpsteer_helpfulness import compute_score

score = compute_score(
    'helpsteer2', response, '', extra_info={'question': prompt}
)
```

## Files

```
recipe/amo_helpsteer/
├── __init__.py
├── reward.proto              # gRPC schema (5 attribute ScoreResponse)
├── compile.sh                # regenerate *_pb2 files
├── reward_pb2.py             # generated
├── reward_pb2_grpc.py        # generated
├── reward_server.py          # ArmoRM gRPC server
├── helpsteer_client.py       # client helpers
├── helpsteer_helpfulness.py  # per-dimension metric wrappers
├── helpsteer_correctness.py
├── helpsteer_coherence.py
├── helpsteer_complexity.py
├── helpsteer_verbosity.py
├── start_server.sh
└── README.md
```
