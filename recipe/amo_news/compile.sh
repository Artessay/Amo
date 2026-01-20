#!/bin/bash

# `pip install grpcio-tools` if you don't have it already
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. recipe/amo_news/summarization.proto
