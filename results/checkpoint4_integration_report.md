# Checkpoint 4 — Integration, Testing and Load Test Report

## 1. Overview

The LearnSmart hybrid recommendation engine was integrated into a FastAPI REST API serving layer. The API provides health monitoring and personalized course recommendations using the best-performing hybrid configuration from Checkpoint 3.

The selected hybrid configuration uses a content-based filtering weight of 0.3 and a collaborative filtering weight of 0.7.

## 2. API Integration

The recommendation model was integrated into a FastAPI REST API with the following endpoints:

* `/` — Returns API status and version information.
* `/health` — Returns API health status, dataset information, and model configuration.
* `/recommend` — Generates personalized course recommendations for a given learner.

### Model Configuration

* Content-based filtering weight: **0.3**
* Collaborative filtering weight: **0.7**
* Recommendation count: Configurable from 1 to 20 courses

The API was tested locally using the LearnSmart dataset containing **100 users and 20 courses**.

## 3. Integration Testing

Integration tests were implemented using **Pytest** and **FastAPI TestClient**.

The following scenarios were tested:

1. Root endpoint availability
2. Health endpoint validation
3. Recommendation endpoint functionality
4. Invalid user handling

### Test Results

* Tests executed: **4**
* Tests passed: **4**
* Tests failed: **0**

**Integration Test Result: PASSED**

## 4. Load Testing

A concurrent load test was performed against the `/recommend` endpoint using Python's `ThreadPoolExecutor` and the `requests` library.

### Load Test Configuration

* Total requests: **50**
* Concurrent requests: **50**
* Test user: **U001**
* Recommendations requested per request: **5**
* Endpoint tested: **POST /recommend**

### Observed Performance Results

| Metric              |               Result |
| ------------------- | -------------------: |
| Total requests      |                   50 |
| Successful requests |                   50 |
| Failed requests     |                    0 |
| Total test time     |       5.2831 seconds |
| Average latency     |       4.3535 seconds |
| Minimum latency     |       3.3130 seconds |
| Maximum latency     |       5.1795 seconds |
| Throughput          | 9.46 requests/second |

**Load Test Result: PASSED**

All 50 concurrent requests were successfully processed without failures.

## 5. Performance Observation

Under the local test environment, the API achieved an average response latency of **4.3535 seconds** and a maximum observed latency of **5.1795 seconds**.

The measured throughput was **9.46 requests per second** while processing 50 concurrent requests.

These measurements represent the performance of the locally running development environment and may vary depending on hardware, network configuration, deployment environment, dataset size, and production infrastructure.

## 6. Conclusion

Checkpoint 4 integration, testing, and load-testing requirements were successfully completed.

The LearnSmart hybrid recommendation model was integrated into a FastAPI REST API and verified through automated integration tests. All **4 integration tests passed** successfully.

The API was also tested with **50 concurrent requests**, achieving **50 successful responses with zero failures**. The observed average latency was **4.3535 seconds**, with a throughput of **9.46 requests per second**.

The implementation demonstrates that the LearnSmart recommendation engine can be served through a REST API and tested for concurrent request handling.

## 7. Files Added

The following files were added or updated for Checkpoint 4:

* `src/api.py` — FastAPI serving layer
* `tests/test_api.py` — API integration tests
* `pytest.ini` — Pytest configuration
* `load_test.py` — Concurrent load-testing script
* `results/checkpoint4_integration_report.md` — Checkpoint 4 performance report
