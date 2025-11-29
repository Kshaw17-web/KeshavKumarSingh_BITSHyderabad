"""
Test script for concurrent requests to measure latency and throughput.
Runs 5 concurrent requests and prints latencies.
"""

import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import statistics

# Configuration
API_URL = "http://localhost:8000/api/v1/hackrx/run"
TEST_URL = "https://example.com/sample-bill.pdf"  # Replace with actual test URL
NUM_REQUESTS = 5
CONCURRENT_WORKERS = 5


def make_request_with_url(request_id: int, url: str, api_url: str) -> Tuple[int, float, bool, Optional[str]]:
    """
    Make a single API request and measure latency.
    
    Returns:
        Tuple of (request_id, latency_seconds, success, error_message)
    """
    start_time = time.time()
    
    try:
        payload = {"document": url}
        response = requests.post(api_url, json=payload, timeout=60)
        
        latency = time.time() - start_time
        success = response.status_code == 200
        
        if success:
            return (request_id, latency, True, None)
        else:
            return (request_id, latency, False, f"HTTP {response.status_code}: {response.text}")
    
    except requests.exceptions.Timeout:
        latency = time.time() - start_time
        return (request_id, latency, False, "Request timeout")
    
    except Exception as e:
        latency = time.time() - start_time
        return (request_id, latency, False, str(e))


def run_concurrent_test(test_url: str, api_url: str, num_requests: int = 5, concurrent: int = 5):
    """
    Run concurrent requests and print statistics.
    
    Args:
        test_url: URL to test
        api_url: API endpoint URL
        num_requests: Number of requests to make
        concurrent: Number of concurrent workers
    """
    print(f"\n{'='*60}")
    print(f"Concurrent Request Test")
    print(f"{'='*60}")
    print(f"API URL: {api_url}")
    print(f"Test URL: {test_url}")
    print(f"Total requests: {num_requests}")
    print(f"Concurrent workers: {concurrent}")
    print(f"{'='*60}\n")
    
    # Check API health
    try:
        health_response = requests.get(api_url.replace("/api/v1/hackrx/run", "/"), timeout=5)
        if health_response.status_code == 200:
            print("✓ API is healthy\n")
        else:
            print(f"⚠ API health check returned {health_response.status_code}\n")
    except Exception as e:
        print(f"⚠ API health check failed: {e}\n")
    
    # Run concurrent requests
    start_time = time.time()
    results: List[Tuple[int, float, bool, Optional[str]]] = []
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(make_request_with_url, i, test_url, api_url)
            for i in range(num_requests)
        ]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            request_id, latency, success, error = result
            status = "✓" if success else "✗"
            print(f"Request {request_id}: {status} {latency:.3f}s", end="")
            if error:
                print(f" ({error[:50]})", end="")
            print()
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    latencies = [r[1] for r in results]
    successes = [r[2] for r in results]
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        if len(latencies) > 1:
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max_latency
        else:
            p95_latency = avg_latency
            p99_latency = avg_latency
        
        success_rate = sum(successes) / len(successes) * 100
        throughput = len(results) / total_time
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total requests: {len(results)}")
        print(f"Successful: {sum(successes)} ({success_rate:.1f}%)")
        print(f"Failed: {len(results) - sum(successes)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"\nLatency Statistics:")
        print(f"  Average: {avg_latency:.3f}s")
        print(f"  Median: {median_latency:.3f}s")
        print(f"  Min: {min_latency:.3f}s")
        print(f"  Max: {max_latency:.3f}s")
        print(f"  P95: {p95_latency:.3f}s")
        print(f"  P99: {p99_latency:.3f}s")
        print(f"{'='*60}\n")
        
        # Check metrics endpoint
        try:
            metrics_response = requests.get(api_url.replace("/api/v1/hackrx/run", "/metrics"), timeout=5)
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                print("API Metrics:")
                print(f"  Requests served: {metrics.get('requests_served', 0)}")
                print(f"  Avg latency: {metrics.get('avg_latency_seconds', 0):.3f}s")
                print(f"  P95 latency: {metrics.get('p95_latency_seconds', 0):.3f}s")
                print(f"  P99 latency: {metrics.get('p99_latency_seconds', 0):.3f}s")
                print(f"  Cache hit rate: {metrics.get('cache_hit_rate', 0):.1%}")
                print(f"  Timeout errors: {metrics.get('timeout_errors', 0)}")
                print(f"  Requests/sec: {metrics.get('requests_per_second', 0):.2f}")
                print()
        except Exception as e:
            print(f"⚠ Could not fetch metrics: {e}\n")
    else:
        print("\n⚠ No results to analyze")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test concurrent API requests")
    parser.add_argument("--url", type=str, default=TEST_URL,
                       help="Test document URL")
    parser.add_argument("--api-url", type=str, default=API_URL,
                       help="API endpoint URL")
    parser.add_argument("--num-requests", type=int, default=NUM_REQUESTS,
                       help="Number of requests to make")
    parser.add_argument("--concurrent", type=int, default=CONCURRENT_WORKERS,
                       help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    run_concurrent_test(args.url, args.api_url, args.num_requests, args.concurrent)


if __name__ == "__main__":
    main()

