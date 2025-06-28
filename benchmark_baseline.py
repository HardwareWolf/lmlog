#!/usr/bin/env python3

import asyncio
import time
import tracemalloc
from statistics import mean, stdev
import tempfile
import os

from lmlog import OptimizedLLMLogger, LLMLogger


def benchmark_sync_logging(logger_class, iterations=10000):
    """Benchmark synchronous logging performance."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    if logger_class == OptimizedLLMLogger:
        logger = logger_class(temp_path, encoder="msgspec")
    else:
        logger = logger_class(temp_path)
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    for i in range(iterations):
        logger.log_event(
            event_type="performance_test",
            entity_type="test",
            entity_id=f"test_{i}",
            context={"iteration": i, "data": "test_data"}
        )
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    if hasattr(logger, 'close'):
        logger.close()
    elif hasattr(logger, '_flush'):
        logger._flush()
    os.unlink(temp_path)
    
    total_time = end_time - start_time
    ops_per_second = iterations / total_time
    time_per_op = (total_time / iterations) * 1_000_000  # microseconds
    
    return {
        "total_time": total_time,
        "ops_per_second": ops_per_second,
        "time_per_op_us": time_per_op,
        "peak_memory_mb": peak / 1024 / 1024,
        "current_memory_mb": current / 1024 / 1024
    }


async def benchmark_async_logging(logger_class, iterations=10000):
    """Benchmark asynchronous logging performance."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    if logger_class == OptimizedLLMLogger:
        logger = logger_class(temp_path, encoder="msgspec")
    else:
        logger = logger_class(temp_path)
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    if hasattr(logger, 'alog_event'):
        tasks = []
        for i in range(iterations):
            task = logger.alog_event(
                event_type="performance_test",
                entity_type="test",
                entity_id=f"test_{i}",
                context={"iteration": i, "data": "test_data"}
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    else:
        for i in range(iterations):
            logger.log_event(
                event_type="performance_test",
                entity_type="test",
                entity_id=f"test_{i}",
                context={"iteration": i, "data": "test_data"}
            )
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    if hasattr(logger, 'close'):
        logger.close()
    elif hasattr(logger, '_flush'):
        logger._flush()
    os.unlink(temp_path)
    
    total_time = end_time - start_time
    ops_per_second = iterations / total_time
    time_per_op = (total_time / iterations) * 1_000_000  # microseconds
    
    return {
        "total_time": total_time,
        "ops_per_second": ops_per_second,
        "time_per_op_us": time_per_op,
        "peak_memory_mb": peak / 1024 / 1024,
        "current_memory_mb": current / 1024 / 1024
    }


def run_multiple_benchmarks(benchmark_func, logger_class, iterations=10000, runs=5):
    """Run multiple benchmark iterations and calculate statistics."""
    results = []
    for _ in range(runs):
        if asyncio.iscoroutinefunction(benchmark_func):
            result = asyncio.run(benchmark_func(logger_class, iterations))
        else:
            result = benchmark_func(logger_class, iterations)
        results.append(result)
    
    stats = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        stats[key] = {
            "mean": mean(values),
            "stdev": stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values)
        }
    
    return stats


def main():
    """Run baseline performance benchmarks."""
    print("=== LMLog Baseline Performance Benchmarks ===\n")
    
    iterations = 50000
    runs = 5
    
    print(f"Running {runs} benchmark runs with {iterations} operations each...\n")
    
    # Benchmark LLMLogger (base class)
    print("1. LLMLogger (Base Class) - Synchronous")
    base_sync_stats = run_multiple_benchmarks(
        benchmark_sync_logging, LLMLogger, iterations, runs
    )
    print(f"   Ops/sec: {base_sync_stats['ops_per_second']['mean']:.0f} ± {base_sync_stats['ops_per_second']['stdev']:.0f}")
    print(f"   Time/op: {base_sync_stats['time_per_op_us']['mean']:.2f} ± {base_sync_stats['time_per_op_us']['stdev']:.2f} μs")
    print(f"   Peak Memory: {base_sync_stats['peak_memory_mb']['mean']:.2f} ± {base_sync_stats['peak_memory_mb']['stdev']:.2f} MB")
    
    print("\n2. LLMLogger (Base Class) - Asynchronous")
    base_async_stats = run_multiple_benchmarks(
        benchmark_async_logging, LLMLogger, iterations, runs
    )
    print(f"   Ops/sec: {base_async_stats['ops_per_second']['mean']:.0f} ± {base_async_stats['ops_per_second']['stdev']:.0f}")
    print(f"   Time/op: {base_async_stats['time_per_op_us']['mean']:.2f} ± {base_async_stats['time_per_op_us']['stdev']:.2f} μs")
    print(f"   Peak Memory: {base_async_stats['peak_memory_mb']['mean']:.2f} ± {base_async_stats['peak_memory_mb']['stdev']:.2f} MB")
    
    # Benchmark OptimizedLLMLogger
    print("\n3. OptimizedLLMLogger - Synchronous")
    opt_sync_stats = run_multiple_benchmarks(
        benchmark_sync_logging, OptimizedLLMLogger, iterations, runs
    )
    print(f"   Ops/sec: {opt_sync_stats['ops_per_second']['mean']:.0f} ± {opt_sync_stats['ops_per_second']['stdev']:.0f}")
    print(f"   Time/op: {opt_sync_stats['time_per_op_us']['mean']:.2f} ± {opt_sync_stats['time_per_op_us']['stdev']:.2f} μs")
    print(f"   Peak Memory: {opt_sync_stats['peak_memory_mb']['mean']:.2f} ± {opt_sync_stats['peak_memory_mb']['stdev']:.2f} MB")
    
    print("\n4. OptimizedLLMLogger - Asynchronous")
    opt_async_stats = run_multiple_benchmarks(
        benchmark_async_logging, OptimizedLLMLogger, iterations, runs
    )
    print(f"   Ops/sec: {opt_async_stats['ops_per_second']['mean']:.0f} ± {opt_async_stats['ops_per_second']['stdev']:.0f}")
    print(f"   Time/op: {opt_async_stats['time_per_op_us']['mean']:.2f} ± {opt_async_stats['time_per_op_us']['stdev']:.2f} μs")
    print(f"   Peak Memory: {opt_async_stats['peak_memory_mb']['mean']:.2f} ± {opt_async_stats['peak_memory_mb']['stdev']:.2f} MB")
    
    print("\n=== Performance Comparison ===")
    opt_sync_improvement = (opt_sync_stats['ops_per_second']['mean'] / base_sync_stats['ops_per_second']['mean']) - 1
    opt_async_improvement = (opt_async_stats['ops_per_second']['mean'] / base_async_stats['ops_per_second']['mean']) - 1
    
    print(f"OptimizedLLMLogger vs LLMLogger (Sync): {opt_sync_improvement:.1%} faster")
    print(f"OptimizedLLMLogger vs LLMLogger (Async): {opt_async_improvement:.1%} faster")
    
    # Save baseline results
    import json
    baseline_results = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": iterations,
        "runs": runs,
        "results": {
            "llm_logger_sync": base_sync_stats,
            "llm_logger_async": base_async_stats,
            "optimized_logger_sync": opt_sync_stats,
            "optimized_logger_async": opt_async_stats
        }
    }
    
    with open("baseline_performance.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    print("\nBaseline results saved to baseline_performance.json")


if __name__ == "__main__":
    main()