import logging
from concurrent import futures
import grpc
import traceback

import summarization_pb2
import summarization_pb2_grpc

try:
    # UniEval entry points
    from recipe.amo_news.metric.evaluator import get_evaluator
    from recipe.amo_news.common import evaluate_dimension
except ImportError as e:  # pragma: no cover - optional runtime dependency
    raise ImportError(
        "UniEval is required for CNN/DailyMail summarization metrics. "
        "Please install it via `pip install -r Amo/requirements.txt`."
    ) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationEvaluationServiceServicer(summarization_pb2_grpc.SummarizationEvaluationServiceServicer):
    """gRPC servicer that wraps the UniEval summarization evaluator."""

    def __init__(self):
        self.evaluator = get_evaluator("summarization")

    def EvaluateSummarization(self, request, context):
        """Evaluate a summarization based on the requested dimension."""
        
        logger.info(
            f"Received evaluation request: dimension={request.dimension}, "
            f"solution_str={request.solution_str[:100]}..."
        )
        
        # Prepare extra_info based on the dimension
        extra_info = {}
        if request.dimension in {"coherence", "consistency"}:
            extra_info["article"] = request.article
        
        # Call the existing evaluation function
        try:
            score = evaluate_dimension(
                evaluator=self.evaluator,
                solution_str=request.solution_str,
                ground_truth=request.ground_truth,
                extra_info=extra_info,
                dim=request.dimension
            )
            
            logger.info(f"Evaluation completed: dimension={request.dimension}, score={score}")
            return summarization_pb2.EvaluationResponse(score=score)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Evaluation failed: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return summarization_pb2.EvaluationResponse(score=0.0)


def serve(port: int = 50053) -> None:
    """Start the summarization evaluation gRPC server."""
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    summarization_pb2_grpc.add_SummarizationEvaluationServiceServicer_to_server(
        SummarizationEvaluationServiceServicer(),
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    logger.info("Starting SummarizationEvaluationService on port %d", port)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the Summarization Evaluation gRPC server.")
    parser.add_argument("--port", type=int, default=50053, help="Port to run the server on.")
    args = parser.parse_args()
    
    serve(port=args.port)
