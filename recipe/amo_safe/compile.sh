
# `pip install grpcio-tools` if you don't have it already
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. reward.proto