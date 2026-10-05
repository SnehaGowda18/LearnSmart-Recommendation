# LearnSmart Recommendation Engine — Model Card

## 1. Model Overview

The LearnSmart Recommendation Engine is a hybrid course recommendation system designed for an EdTech learning platform.

The system recommends relevant next courses to learners based on their previous course completion history, quiz performance, course characteristics, and similarities with other learners.

The final recommendation system combines:

* Content-Based Filtering
* Collaborative Filtering
* Hybrid Score Ranking

The best-performing configuration uses:

* Content-Based Weight: 0.3
* Collaborative Filtering Weight: 0.7

The recommendation engine is exposed through a FastAPI REST API.

---

## 2. Intended Use

The system is intended to help learners discover suitable next courses based on their learning history.

Potential use cases include:

* Personalized course recommendations
* Adaptive learning paths
* Next-course prediction
* Learning content discovery
* Supporting learner engagement

The system is designed as a recommendation-support tool and should not be used as the sole basis for academic or high-impact decisions.

---

## 3. Dataset

The project uses the LearnSmart interaction dataset containing:

* 100 users
* 20 courses
* 900 interaction records

The dataset contains information including:

* User ID
* Course ID
* Course name
* Course category
* Course level
* Learning sequence
* Completion status
* Quiz score
* Rating
* Relevance label

The dataset was prepared for experimentation and recommendation-system evaluation.

---

## 4. Recommendation Approach

### Content-Based Filtering

The content-based component represents courses using their textual and categorical characteristics.

TF-IDF features and cosine similarity are used to identify courses that are similar to the learner's previous learning interests.

Completed courses are excluded from the final recommendation list.

### Collaborative Filtering

The collaborative component builds a learner-course interaction representation using:

* Course completion
* Quiz performance
* Similar learner behavior

Cosine similarity is used to identify learners with similar interaction patterns.

Courses associated with similar learners are then considered for recommendation.

### Hybrid Recommendation

The final recommendation score combines the normalized content-based and collaborative scores.

The final selected configuration is:

```text
Content-Based Weight       = 0.3
Collaborative Weight       = 0.7
```

This configuration produced the strongest validation performance during experimentation.

---

## 5. Evaluation Metrics

The primary evaluation metric is Precision@5.

Precision@5 measures how many of the top five recommended courses are relevant to the learner.

### Final Evaluation Result

| Configuration   | Content Weight | Collaborative Weight | Precision@5 |
| --------------- | -------------: | -------------------: | ----------: |
| Configuration A |            0.3 |                  0.7 |      0.9440 |
| Configuration B |            0.7 |                  0.3 |      0.6980 |

The best configuration achieved:

**Precision@5 = 0.9440 (94.40%)**

The project requirement was:

**Precision@5 >= 0.60 (60%)**

Therefore, the best-performing configuration exceeded the required evaluation target.

---

## 6. Cold-Start and Warm-Start Handling

### Warm Start

For existing learners with interaction history, the system uses completed courses, quiz scores, and learner similarity to generate personalized recommendations.

### Cold Start

For learners with limited or no historical interaction data, recommendations can rely more heavily on course-content similarity and available course metadata.

Cold-start behavior should be further validated with larger real-world learner datasets before production deployment.

---

## 7. API and Deployment

The recommendation engine is served using FastAPI.

### API Endpoints

#### GET `/`

Returns API status and version information.

#### GET `/health`

Returns:

* API health status
* Number of users
* Number of courses
* Model type
* Hybrid model weights

#### POST `/recommend`

Accepts a learner ID and requested recommendation count and returns personalized course recommendations.

Example request:

```json
{
  "user_id": "U001",
  "top_n": 5
}
```

### Local Deployment

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the API from the project root:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

The API can then be accessed locally through:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## 8. Testing

Automated API integration tests were implemented using Pytest and FastAPI TestClient.

The test suite validates:

1. Root endpoint
2. Health endpoint
3. Recommendation endpoint
4. Invalid user handling

### Test Result

```text
4 tests passed
0 tests failed
```

The integration test suite passed successfully.

---

## 9. Load Testing

The API was tested with 50 concurrent requests against the `/recommend` endpoint.

### Observed Results

| Metric              |               Result |
| ------------------- | -------------------: |
| Total requests      |                   50 |
| Concurrent requests |                   50 |
| Successful requests |                   50 |
| Failed requests     |                    0 |
| Total test time     |       5.2831 seconds |
| Average latency     |       4.3535 seconds |
| Minimum latency     |       3.3130 seconds |
| Maximum latency     |       5.1795 seconds |
| Throughput          | 9.46 requests/second |

All 50 requests completed successfully.

These measurements were obtained from the local development environment and should not be treated as production capacity benchmarks.

---

## 10. Bias and Fairness Audit

The current dataset does not contain sensitive demographic attributes such as gender, caste, religion, ethnicity, or other protected characteristics.

The recommendation model primarily uses learning behavior and course-related information.

However, recommendation bias can still occur through behavioral patterns.

Potential sources of bias include:

* Over-recommending popular courses
* Reinforcing existing learning preferences
* Limited representation of some course categories
* Sparse interaction history for some learners
* Synthetic dataset characteristics

### Mitigation

Potential mitigation strategies include:

* Monitoring recommendation distribution across course categories
* Evaluating coverage and diversity
* Monitoring cold-start performance
* Periodically reviewing recommendation quality
* Testing with larger and more representative real-world datasets

A more comprehensive fairness audit should be performed before production deployment using appropriate real-world demographic and behavioral data where legally and ethically permitted.

---

## 11. Limitations

The current system has several limitations:

1. The dataset is synthetic and may not fully represent real learner behavior.
2. The evaluation dataset is relatively small.
3. Production-scale traffic was not evaluated.
4. Cold-start performance requires additional validation.
5. Recommendation quality may change as learner behavior changes.
6. The current system does not include real-time model retraining.
7. Latency measured during local testing may differ from production latency.
8. The model has not been evaluated against a large production benchmark.

---

## 12. Reproducibility

The project source code, dataset, evaluation results, API implementation, tests, and load-testing script are maintained in the GitHub repository.

Repository:

```text
https://github.com/SnehaGowda18/LearnSmart-Recommendation
```

Important project files include:

```text
src/
├── api.py
├── hybrid_model.py
├── collaborative_model.py
├── baseline_model.py
└── data_pipeline.py

tests/
└── test_api.py

results/
├── baseline_report.md
├── checkpoint3_experiment_report.md
├── checkpoint4_integration_report.md
├── hyperparameter_results.csv
├── hybrid_weight_experiment.csv
└── model_card.md

load_test.py
pytest.ini
requirements.txt
```

---

## 13. Final Status

The LearnSmart Recommendation Engine successfully integrates the hybrid recommendation model with a REST API.

Key achievements:

* Hybrid recommendation model implemented
* Content-based filtering implemented
* Collaborative filtering implemented
* Best Precision@5: **94.40%**
* Required Precision@5: **60%**
* FastAPI serving layer implemented
* API integration tests: **4/4 passed**
* Concurrent load test: **50 requests**
* Successful requests: **50/50**
* Failed requests: **0**
* Average latency: **4.3535 seconds**
* Throughput: **9.46 requests/second**

The system is ready for final demonstration and further production-oriented validation.
