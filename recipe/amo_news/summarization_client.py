import os
import grpc

from recipe.amo_news import summarization_pb2, summarization_pb2_grpc

def evaluate_summarization(solution_str: str, ground_truth: str, extra_info: dict | None, dim: str) -> float:
    """Evaluate a summarization dimension using the gRPC service.
    
    Args:
        solution_str: Model-generated summary.
        ground_truth: Reference summary (used by 'relevance'; ignored otherwise).
        extra_info: Optional metadata; for 'coherence' and 'consistency' this
            must contain an 'article' field with the source document.
        dim: One of {"coherence", "consistency", "fluency", "relevance"}.
    
    Returns:
        The evaluation score.
    """
    
    # Get server address from environment variables
    host = os.getenv('NEWS_TARGET_HOST', 'localhost')
    port = os.getenv('NEWS_TARGET_PORT', '50053')
    
    # Prepare article content if needed
    article = ""
    if dim in {"coherence", "consistency"} and isinstance(extra_info, dict):
        article = extra_info.get("article", "")
    
    # Create gRPC channel and stub
    with grpc.insecure_channel(f'{host}:{port}') as channel:
        stub = summarization_pb2_grpc.SummarizationEvaluationServiceStub(channel)
        
        # Create request message
        request = summarization_pb2.EvaluationRequest(
            solution_str=solution_str,
            ground_truth=ground_truth,
            article=article,
            dimension=dim
        )
        
        # Make the RPC call
        response = stub.EvaluateSummarization(request)
        # response = stub.EvaluateSummarization(request, timeout=30.0)
        return response.score

if __name__ == "__main__":
    # Example usage
    article = "This is a sample article about technology. It discusses the latest advancements in artificial intelligence and their impact on society."
    solution_str = "AI technology is advancing rapidly and affecting various industries."
    ground_truth = "Artificial intelligence is making significant progress and impacting society."
    
    extra_info = {"article": article}
    
    # Test all dimensions
    dimensions = ["coherence", "consistency", "fluency", "relevance"]
    
    print("Testing summarization evaluation service...")
    print(f"Article: {article[:100]}...")
    print(f"Solution: {solution_str}")
    print(f"Ground truth: {ground_truth}")
    print("=" * 60)
    
    for dim in dimensions:
        try:
            score = evaluate_summarization(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info if dim in {"coherence", "consistency"} else None,
                dim=dim
            )
            print(f"{dim}: {score:.4f}")
        except Exception as e:
            print(f"Error evaluating {dim}: {e}")
    
    print("=" * 60)
    print("Test completed.")
