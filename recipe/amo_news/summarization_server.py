import logging
from concurrent import futures
import grpc
import traceback
import queue
import torch

import summarization_pb2
import summarization_pb2_grpc

try:
    # UniEval entry points
    from recipe.amo_news.metric.evaluator import get_evaluator
    from recipe.amo_news.common import evaluate_dimension
except ImportError as e:
    raise ImportError(
        "UniEval is required for CNN/DailyMail summarization metrics. "
        "Please install it via `pip install -r Amo/requirements.txt`."
    ) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_WORKERS_PER_DEVICE = 1

class SummarizationEvaluationServiceServicer(summarization_pb2_grpc.SummarizationEvaluationServiceServicer):
    """gRPC servicer that wraps the UniEval summarization evaluator with multi-GPU support."""

    def __init__(self):
        # Detect available GPUs
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count == 0:
            logger.warning("No GPUs detected. Falling back to a single CPU instance.")
            self.gpu_count = 1
            devices = ["cpu"]
        else:
            logger.info(f"Detected {self.gpu_count} GPUs. Initializing model pool...")
            devices = [f"cuda:{i}" for i in range(self.gpu_count)] * NUM_WORKERS_PER_DEVICE

        # Create a thread-safe queue to manage the pool of evaluators
        self.evaluator_pool = queue.Queue()

        # Load one evaluator instance per GPU
        for i, device in enumerate(devices):
            try:
                logger.info(f"Loading evaluator {i+1}/{len(devices)} on {device}...")
                
                # Use context manager to ensure the model is loaded onto the correct GPU
                evaluator = get_evaluator("summarization", device=device)

                # Store both the model and its device info in the pool
                self.evaluator_pool.put({
                    "model": evaluator,
                    "device": device
                })
            except Exception as e:
                logger.error(f"Failed to load evaluator on {device}: {e}")
                traceback.print_exc()

        if self.evaluator_pool.empty():
            raise RuntimeError("Critical Error: Could not initialize any evaluators.")
        
        logger.info(f"Service initialized successfully with {self.evaluator_pool.qsize()} evaluators.")

    def EvaluateSummarization(self, request, context):
        """Evaluate a summarization based on the requested dimension using an available GPU."""
        
        # 1. Acquire an evaluator from the pool (blocks if all GPUs are busy)
        resource = self.evaluator_pool.get()
        evaluator = resource["model"]
        device = resource["device"]

        try:
            logger.debug(
                f"Processing request on {device}: dimension={request.dimension}, "
                f"solution_str_preview={request.solution_str[:50]}..."
            )
            
            extra_info = {}
            if request.dimension in {"coherence", "consistency"}:
                extra_info["article"] = request.article
            
            # 2. Perform the evaluation
            score = evaluate_dimension(
                evaluator=evaluator,
                solution_str=request.solution_str,
                ground_truth=request.ground_truth,
                extra_info=extra_info,
                dim=request.dimension
            )
            
            logger.info(f"Evaluation completed on {device}: score={score}")
            return summarization_pb2.EvaluationResponse(score=score)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Evaluation failed on {device}: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return summarization_pb2.EvaluationResponse(score=0.0)
        
        finally:
            # 3. Always return the evaluator to the pool, regardless of success or failure
            self.evaluator_pool.put(resource)


def serve(port: int = 50053) -> None:
    """Start the summarization evaluation gRPC server with high-concurrency support."""
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: 
        num_gpus = 1
    
    # Set max_workers higher than GPU count to handle I/O and queuing overhead
    max_workers = num_gpus * NUM_WORKERS_PER_DEVICE
    
    logger.info(f"Starting server on port {port} with max_workers={max_workers}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    summarization_pb2_grpc.add_SummarizationEvaluationServiceServicer_to_server(
        SummarizationEvaluationServiceServicer(),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("Server is running. Waiting for termination...")
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the Summarization Evaluation gRPC server.")
    parser.add_argument("--port", type=int, default=50053, help="Port to run the server on.")
    args = parser.parse_args()
    
    serve(port=args.port)