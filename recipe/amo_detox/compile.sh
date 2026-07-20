
# `pip install grpcio-tools` if you don't have it already
# Run from the workspace root (Amo/).
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. recipe/amo_detox/reward.proto
