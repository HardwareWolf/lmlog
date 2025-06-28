#!/usr/bin/env python3

import asyncio
import time
import tracemalloc
from statistics import mean, stdev
import tempfile
import os

from lmlog import LLMLogger


def benchmark_sync_logging(async_processing=False, iterations=10000):
    """Benchmark synchronous logging performance."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    logger = LLMLogger(temp_path, async_processing=async_processing, encoder="msgspec")
    
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
        if asyncio.iscoroutinefunction(logger.close):
            asyncio.run(logger.close())
        else:
            logger.close()
    
    os.unlink(temp_path)
    
    total_time = end_time - start_time
    ops_per_second = iterations / total_time
    time_per_op = (total_time / iterations) * 1_000_000
    
    return {
        "total_time": total_time,
        "ops_per_second": ops_per_second,
        "time_per_op_us": time_per_op,
        "peak_memory_mb": peak / 1024 / 1024,
        "current_memory_mb": current / 1024 / 1024
    }


async def benchmark_async_logging(async_processing=True, iterations=10000):
    """Benchmark asynchronous logging performance."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    logger = LLMLogger(temp_path, async_processing=async_processing, encoder="msgspec")
    
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
        await logger.close()
    
    os.unlink(temp_path)
    
    total_time = end_time - start_time
    ops_per_second = iterations / total_time
    time_per_op = (total_time / iterations) * 1_000_000
    
    return {
        "total_time": total_time,
        "ops_per_second": ops_per_second,
        "time_per_op_us": time_per_op,
        "peak_memory_mb": peak / 1024 / 1024,
        "current_memory_mb": current / 1024 / 1024
    }


def run_multiple_benchmarks(benchmark_func, async_processing=False, iterations=10000, runs=5):
    """Run multiple benchmark iterations and calculate statistics."""
    results = []
    for _ in range(runs):
        if asyncio.iscoroutinefunction(benchmark_func):
            result = asyncio.run(benchmark_func(async_processing, iterations))
        else:
            result = benchmark_func(async_processing, iterations)
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
    """Run LLMLogger performance benchmarks."""
    print("=== LMLog Performance Benchmarks ===\n")
    
    iterations = 50000
    runs = 5
    
    print(f"Running {runs} benchmark runs with {iterations} operations each...\n")
    
    print("1. LLMLogger (Sync Mode) - Synchronous")
    sync_mode_stats = run_multiple_benchmarks(
        benchmark_sync_logging, False, iterations, runs
    )
    print(f"   Ops/sec: {sync_mode_stats['ops_per_second']['mean']:.0f} ± {sync_mode_stats['ops_per_second']['stdev']:.0f}")
    print(f"   Time/op: {sync_mode_stats['time_per_op_us']['mean']:.2f} ± {sync_mode_stats['time_per_op_us']['stdev']:.2f} μs")
    print(f"   Peak Memory: {sync_mode_stats['peak_memory_mb']['mean']:.2f} ± {sync_mode_stats['peak_memory_mb']['stdev']:.2f} MB")
    
    print("\n2. LLMLogger (Async Mode) - Synchronous")
    async_sync_stats = run_multiple_benchmarks(
        benchmark_sync_logging, True, iterations, runs
    )
    print(f"   Ops/sec: {async_sync_stats['ops_per_second']['mean']:.0f} ± {async_sync_stats['ops_per_second']['stdev']:.0f}")
    print(f"   Time/op: {async_sync_stats['time_per_op_us']['mean']:.2f} ± {async_sync_stats['time_per_op_us']['stdev']:.2f} μs")
    print(f"   Peak Memory: {async_sync_stats['peak_memory_mb']['mean']:.2f} ± {async_sync_stats['peak_memory_mb']['stdev']:.2f} MB")
    
    print("\n3. LLMLogger (Async Mode) - Asynchronous")
    async_async_stats = run_multiple_benchmarks(
        benchmark_async_logging, True, iterations, runs
    )
    print(f"   Ops/sec: {async_async_stats['ops_per_second']['mean']:.0f} ± {async_async_stats['ops_per_second']['stdev']:.0f}")
    print(f"   Time/op: {async_async_stats['time_per_op_us']['mean']:.2f} ± {async_async_stats['time_per_op_us']['stdev']:.2f} μs")
    print(f"   Peak Memory: {async_async_stats['peak_memory_mb']['mean']:.2f} ± {async_async_stats['peak_memory_mb']['stdev']:.2f} MB")
    
    print("\n=== Performance Comparison ===")
    async_improvement = (async_sync_stats['ops_per_second']['mean'] / sync_mode_stats['ops_per_second']['mean']) - 1
    async_async_improvement = (async_async_stats['ops_per_second']['mean'] / sync_mode_stats['ops_per_second']['mean']) - 1
    
    print(f"Async Mode (Sync calls) vs Sync Mode: {async_improvement:.1%}")
    print(f"Async Mode (Async calls) vs Sync Mode: {async_async_improvement:.1%}")
    
    print("\n=== Memory Efficiency ===")
    sync_mem = sync_mode_stats['peak_memory_mb']['mean']
    async_mem = async_sync_stats['peak_memory_mb']['mean']
    
    print(f"Memory change with Async Mode: {((async_mem/sync_mem) - 1):.1%}")
    
    import json
    results = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": iterations,
        "runs": runs,
        "results": {
            "sync_mode": sync_mode_stats,
            "async_mode_sync": async_sync_stats,
            "async_mode_async": async_async_stats
        }
    }
    
    with open("performance.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to performance.json")


if __name__ == "__main__":
    main()