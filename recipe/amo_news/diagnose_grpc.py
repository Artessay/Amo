"""
诊断脚本：排查gRPC超时问题
"""
import os
import time
import grpc
import threading

from recipe.amo_news import summarization_pb2, summarization_pb2_grpc

class DiagnosticClient:
    def __init__(self, host='localhost', port=50053, timeout=30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.channel = None
        self.stub = None

    def connect(self):
        """建立连接并打印连接状态"""
        print(f"Connecting to {self.host}:{self.port}...")
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')
        # 等待通道就绪，最多5秒
        grpc.channel_ready_future(self.channel).result(timeout=5)
        self.stub = summarization_pb2_grpc.SummarizationEvaluationServiceStub(
            self.channel
        )
        print("✓ Connected successfully")

    def test_single_dimension(self, dim: str, solution_str: str,
                             ground_truth: str = "", article: str = "",
                             extra_info: dict = None):
        """测试单个维度，记录详细时间"""
        print(f"\n{'='*60}")
        print(f"Testing dimension: {dim}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            request = summarization_pb2.EvaluationRequest(
                solution_str=solution_str,
                ground_truth=ground_truth if dim == "relevance" else "",
                article=article if dim in {"coherence", "consistency"} else "",
                dimension=dim
            )

            print(f"  Sending request at {time.strftime('%H:%M:%S')}")

            # 使用长超时时间
            response = self.stub.EvaluateSummarization(
                request,
                timeout=self.timeout
            )

            elapsed = time.time() - start_time
            print(f"  ✓ Response received at {time.strftime('%H:%M:%S')}")
            print(f"  ✓ Score: {response.score:.4f}")
            print(f"  ✓ Total time: {elapsed:.2f}s")

            return response.score, elapsed

        except grpc.RpcError as e:
            elapsed = time.time() - start_time
            print(f"  ✗ Error at {time.strftime('%H:%M:%S')}")
            print(f"  ✗ Error code: {e.code()}")
            print(f"  ✗ Error details: {e.details()}")
            print(f"  ✗ Elapsed time: {elapsed:.2f}s")
            return None, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ✗ Unexpected error: {e}")
            print(f"  ✗ Elapsed time: {elapsed:.2f}s")
            return None, elapsed

    def close(self):
        """关闭连接"""
        if self.channel:
            self.channel.close()
            print("\n✓ Connection closed")


def run_diagnostic():
    """运行诊断测试"""
    print("=" * 60)
    print("Summarization Evaluation Service Diagnostic Tool")
    print("=" * 60)

    # 测试数据
    article = """
    Artificial intelligence is rapidly transforming various industries around the world.
    Machine learning algorithms are being applied in healthcare, finance, transportation,
    and many other sectors. Companies are investing billions of dollars in AI research
    and development. The technology promises to increase productivity and solve complex
    problems that were previously unsolvable.
    """

    solution_str = "AI technology is advancing rapidly and affecting various industries through machine learning applications."

    ground_truth = "Artificial intelligence is making significant progress and impacting society across multiple sectors."

    # 创建诊断客户端
    client = DiagnosticClient(timeout=60.0)  # 使用60秒超时

    try:
        client.connect()

        # 测试所有维度（按顺序）
        dimensions = ["coherence", "consistency", "fluency", "relevance"]
        results = {}

        for dim in dimensions:
            score, elapsed = client.test_single_dimension(
                dim=dim,
                solution_str=solution_str,
                ground_truth=ground_truth,
                article=article
            )
            results[dim] = {"score": score, "time": elapsed}

        # 汇总报告
        print("\n" + "=" * 60)
        print("Diagnostic Summary")
        print("=" * 60)
        print(f"{'Dimension':<15} {'Score':<12} {'Time (s)':<10} {'Status'}")
        print("-" * 60)

        for dim, data in results.items():
            status = "✓ PASS" if data["score"] is not None else "✗ FAIL"
            score_str = f"{data['score']:.4f}" if data["score"] else "N/A"
            time_str = f"{data['time']:.2f}" if data["time"] else "N/A"
            print(f"{dim:<15} {score_str:<12} {time_str:<10} {status}")

        # 检查是否有维度超时
        failed_dims = [d for d, r in results.items() if r["score"] is None]
        if failed_dims:
            print(f"\n⚠️  Failed dimensions: {', '.join(failed_dims)}")
            print("Suggestions:")
            print("  1. Increase timeout value")
            print("  2. Increase server worker threads")
            print("  3. Check GPU memory usage")
            print("  4. Check if model is properly loaded")

    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        client.close()


if __name__ == "__main__":
    run_diagnostic()
