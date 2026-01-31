"""
VectorSearch CLI Entrypoint

Commands:
    vectorsearch serve      Start gRPC/REST server
    vectorsearch benchmark  Run performance benchmarks
    vectorsearch version    Show version info
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn


def main() -> NoReturn:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="vectorsearch",
        description="Planetary-Scale Distributed Vector Search & Inference System",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start vector search server")
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=50051,
        help="gRPC server port (default: 50051)",
    )
    serve_parser.add_argument(
        "--rest-port",
        type=int,
        default=8080,
        help="REST gateway port (default: 8080)",
    )
    serve_parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory for index persistence",
    )
    serve_parser.add_argument(
        "--mode",
        choices=["production", "development", "test"],
        default="development",
        help="Server mode (default: development)",
    )
    
    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument(
        "--dataset",
        choices=["sift-1m", "gist-1m", "deep-1m", "random"],
        default="random",
        help="Benchmark dataset",
    )
    bench_parser.add_argument(
        "--vectors",
        type=int,
        default=100000,
        help="Number of vectors for random dataset",
    )
    bench_parser.add_argument(
        "--dimension",
        type=int,
        default=128,
        help="Vector dimension for random dataset",
    )
    bench_parser.add_argument(
        "--queries",
        type=int,
        default=1000,
        help="Number of search queries",
    )
    
    args = parser.parse_args()
    
    if args.command == "serve":
        _run_server(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(0)
    
    sys.exit(0)


def _get_version() -> str:
    """Get package version."""
    try:
        from vector_search import __version__
        return __version__
    except ImportError:
        return "0.0.0-unknown"


def _run_server(args: argparse.Namespace) -> None:
    """Start the vector search server."""
    print(f"Starting VectorSearch server...")
    print(f"  gRPC port: {args.port}")
    print(f"  REST port: {args.rest_port}")
    print(f"  Data dir:  {args.data_dir}")
    print(f"  Mode:      {args.mode}")
    
    # TODO: Implement server startup
    # from vector_search.api.grpc_server import serve
    # serve(port=args.port, rest_port=args.rest_port, data_dir=args.data_dir)
    print("\n[Server implementation pending - Iteration 1]")


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run performance benchmarks."""
    print(f"Running VectorSearch benchmarks...")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Vectors:   {args.vectors}")
    print(f"  Dimension: {args.dimension}")
    print(f"  Queries:   {args.queries}")
    
    # TODO: Implement benchmarks
    # from vector_search.benchmarks import run_recall_benchmark
    # run_recall_benchmark(...)
    print("\n[Benchmark implementation pending - Iteration 1]")


if __name__ == "__main__":
    main()
