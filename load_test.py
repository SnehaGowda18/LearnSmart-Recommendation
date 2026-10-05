import concurrent.futures
import statistics
import time

import requests


API_URL = "http://127.0.0.1:8000/recommend"
TOTAL_REQUESTS = 50
CONCURRENT_REQUESTS = 50


def send_request():
    payload = {
        "user_id": "U001",
        "top_n": 5,
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30,
        )

        latency = time.perf_counter() - start

        return {
            "status_code": response.status_code,
            "latency": latency,
            "success": response.status_code == 200,
        }

    except requests.RequestException:
        latency = time.perf_counter() - start

        return {
            "status_code": 0,
            "latency": latency,
            "success": False,
        }


def main():
    print("LearnSmart API Load Test")
    print("=" * 45)
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    print()

    test_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENT_REQUESTS
    ) as executor:
        results = list(
            executor.map(
                lambda _: send_request(),
                range(TOTAL_REQUESTS),
            )
        )

    total_time = time.perf_counter() - test_start

    latencies = [result["latency"] for result in results]
    successful = sum(result["success"] for result in results)
    failed = TOTAL_REQUESTS - successful

    average_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    throughput = TOTAL_REQUESTS / total_time

    print(f"Successful requests: {successful}")
    print(f"Failed requests: {failed}")
    print(f"Total test time: {total_time:.4f} seconds")
    print(f"Average latency: {average_latency:.4f} seconds")
    print(f"Minimum latency: {min_latency:.4f} seconds")
    print(f"Maximum latency: {max_latency:.4f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")

    print()
    if successful == TOTAL_REQUESTS:
        print("Load test result: PASSED")
    else:
        print("Load test result: FAILED")


if __name__ == "__main__":
    main()