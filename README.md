# LearnSmart Recommendation Engine

An adaptive EdTech course recommendation engine that recommends the next suitable courses for learners using a hybrid recommendation approach combining **content-based filtering** and **collaborative filtering**.

The system uses learner course interactions, completion status, quiz performance, ratings, course categories, levels, and related course information to generate personalized recommendations.

---

## Project Overview

**Project:** LearnSmart EdTech Recommendation Engine
**Domain:** Education Technology / Artificial Intelligence / Machine Learning
**Type:** Hybrid Recommendation System
**API:** FastAPI
**Language:** Python
**Repository:** [LearnSmart-Recommendation](https://github.com/SnehaGowda18/LearnSmart-Recommendation)

### Main Objectives

* Recommend relevant next courses for learners.
* Combine content-based and collaborative recommendation methods.
* Support warm-start and cold-start recommendation scenarios.
* Evaluate recommendations using Precision@5.
* Provide recommendations through a REST API.
* Test the API and system performance.
* Track model experiments.
* Document the final model using a model card.
* Automate testing using GitHub Actions CI.

---

# System Workflow

```text
Learner Interaction Data
          |
          v
     Data Pipeline
          |
          v
  Data Cleaning & Preparation
          |
          +----------------------+
          |                      |
          v                      v
 Content-Based Model     Collaborative Model
          |                      |
          +----------+-----------+
                     |
                     v
             Hybrid Recommender
                     |
                     v
              FastAPI REST API
                     |
                     v
          Personalized Courses
```

---

# Checkpoint 1 — Research & Architecture

## Objective

Research recommendation-system approaches and define the architecture for the LearnSmart adaptive learning recommendation engine.

## Completed Work

* Studied recommendation-system approaches.
* Selected a hybrid recommendation strategy.
* Planned content-based and collaborative filtering components.
* Defined learner-course interaction requirements.
* Defined recommendation evaluation using Precision@5.
* Planned REST API integration.
* Defined cold-start and warm-start scenarios.
* Established the overall project structure and implementation approach.

## Planned Architecture

The system combines:

1. **Content-Based Filtering**

   * Uses course information such as course name, category, and level.
   * Uses TF-IDF vectorization.
   * Calculates course similarity using cosine similarity.

2. **Collaborative Filtering**

   * Uses learner-course interaction information.
   * Considers completion and quiz performance.
   * Uses learner similarity to identify relevant courses.

3. **Hybrid Recommendation**

   * Combines content and collaborative scores.
   * Produces the final ranked course recommendations.

---

# Checkpoint 2 — Data Pipeline & Baseline Model

## Objective

Create the interaction dataset, prepare the data pipeline, and establish a baseline recommendation model.

## Dataset

The final LearnSmart dataset contains:

* **100 learners**
* **20 courses**
* **900 interaction records**

The dataset contains learner-course interaction information including:

* `user_id`
* `course_id`
* `course_name`
* `category`
* `level`
* `completed`
* `quiz_score`
* `rating`

Additional recommendation/evaluation information is used by the project pipeline where required.

## Data Pipeline

The data pipeline performs:

* Dataset loading
* Data cleaning
* Missing-value checking
* Data validation
* Feature preparation
* Interaction preparation for recommendation models

The final cleaned dataset contains **900 records with no missing values**.

## Baseline

A sequence-aware/baseline recommendation approach was developed before the hybrid experiments.

The baseline was used as a reference point for subsequent model improvements.

---

# Checkpoint 3 — Core Model & Experimentation

## Objective

Develop the hybrid recommendation model and evaluate different content/collaborative weight configurations.

## Recommendation Approach

### Content-Based Filtering

Course information is transformed using **TF-IDF vectorization**.

Course similarity is calculated using:

```text
Cosine Similarity
```

This allows the system to identify courses that are similar to courses a learner has interacted with.

### Collaborative Filtering

The collaborative component uses learner-course interactions and combines information such as:

* Course completion
* Quiz performance
* Similar learner behavior

This allows the system to identify courses that may be useful based on patterns from other learners.

### Hybrid Model

The final recommendation score combines the content and collaborative scores.

Best configuration:

```text
Content Weight       = 0.3
Collaborative Weight = 0.7
```

## Experiment Results

### Best Configuration

```text
Content Weight       : 0.3
Collaborative Weight : 0.7
Precision@5          : 0.9440
Precision@5 (%)      : 94.40%
```

### Alternative Configuration

```text
Content Weight       : 0.7
Collaborative Weight : 0.3
Precision@5          : 0.6980
Precision@5 (%)      : 69.80%
```

### Acceptance Target

```text
Required Precision@5 : 0.60
Achieved Precision@5 : 0.9440
```

The best configuration therefore exceeds the required **60% Precision@5 target**.

## Experiment Tracking

Model experiments were tracked using **MLflow** under the experiment:

```text
LearnSmart_Hybrid_Recommender
```

Experiment results and supporting files are available in the `results/` directory.

---

# Checkpoint 4 — Integration & Testing

## Objective

Integrate the recommendation model into a REST API and validate the system through automated testing and load testing.

## FastAPI

The recommendation engine was integrated into a FastAPI application.

### API Endpoints

```text
GET /
```

Returns basic API information.

```text
GET /health
```

Returns API health and model information.

Example information:

```text
Status: healthy
Users: 100
Courses: 20
Model: hybrid
Content Weight: 0.3
Collaborative Weight: 0.7
```

```text
POST /recommend
```

Generates personalized recommendations.

Example request:

```json
{
  "user_id": "U001",
  "top_n": 5
}
```

Example API documentation is available through FastAPI Swagger UI when running locally:

```text
http://127.0.0.1:8005/docs
```

## Automated Testing

The project contains API integration tests covering:

* Root endpoint
* Health endpoint
* Recommendation endpoint
* Invalid-user handling

Test command:

```powershell
pytest -q
```

Final result:

```text
4 passed
```

There were **0 test failures**.

## Load Testing

The API was tested with:

```text
Total Requests       : 50
Concurrent Requests  : 50
Successful Requests  : 50
Failed Requests      : 0
```

Performance results:

```text
Total Test Time      : 5.2831 seconds
Average Latency      : 4.3535 seconds
Minimum Latency      : 3.3130 seconds
Maximum Latency      : 5.1795 seconds
Throughput           : 9.46 requests/second
```

The load test completed successfully with **50/50 successful requests**.

---

# Checkpoint 5 — Final Demo & Model Card

## Objective

Complete the end-to-end system, document the model, validate CI, and demonstrate the final application.

## Model Card

A complete model card is available at:

```text
results/model_card.md
```

The model card documents:

* Model overview
* Intended use
* Dataset
* Recommendation methodology
* Evaluation metrics
* Cold-start behavior
* Warm-start behavior
* API usage
* Deployment instructions
* Testing
* Load testing
* Bias and fairness audit
* Limitations
* Reproducibility
* Repository information

## Bias & Fairness Audit

The current dataset does not contain sensitive demographic attributes.

Potential sources of recommendation bias include:

* Uneven course popularity
* Limited learner interaction history
* Synthetic/limited training data
* Similarity-based recommendation effects
* Cold-start learners

Potential mitigation approaches include:

* Monitoring recommendation distributions
* Expanding learner interaction data
* Evaluating recommendations across different learner groups when demographic data is legitimately available
* Periodically reviewing model performance
* Improving cold-start strategies

## Limitations

The current system has several limitations:

* The interaction dataset is limited in size.
* Cold-start recommendations have limited personalization when learner history is unavailable.
* Collaborative filtering depends on sufficient interaction data.
* The current evaluation uses Precision@5 as the primary metric.
* Real-world production deployment would require larger and continuously updated learner data.
* Recommendation quality may change as course catalogs and learner behavior change.

## Deployment

Install the required Python dependencies and activate the virtual environment.

Start the API locally with:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8005
```

Open the Swagger API documentation:

```text
http://127.0.0.1:8005/docs
```

## CI/CD

The repository includes a GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

The CI workflow:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs required dependencies.
4. Runs the automated test suite.

The final GitHub Actions workflow is passing successfully.

---

# Final Project Results

| Metric                    |            Result |
| ------------------------- | ----------------: |
| Learners                  |               100 |
| Courses                   |                20 |
| Interaction Records       |               900 |
| Best Content Weight       |               0.3 |
| Best Collaborative Weight |               0.7 |
| Precision@5               |        **94.40%** |
| Required Precision@5      |               60% |
| API Tests                 |    **4/4 Passed** |
| Load-Test Requests        |                50 |
| Successful Requests       |         **50/50** |
| Failed Requests           |                 0 |
| Load-Test Throughput      | 9.46 requests/sec |
| CI Status                 |       **Passing** |

---

# Project Structure

```text
LearnSmart-Recommendation/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── learnsmart_interactions.csv
│
├── results/
│   ├── checkpoint3_experiment_report.md
│   ├── checkpoint4_integration_report.md
│   ├── hyperparameter_results.csv
│   ├── hybrid_weight_experiment.csv
│   └── model_card.md
│
├── src/
│   ├── api.py
│   ├── collaborative_model.py
│   ├── content_model.py
│   └── hybrid_model.py
│
├── tests/
│   └── test_api.py
│
├── load_test.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# Testing

Run the complete automated test suite:

```powershell
pytest -q
```

Expected result:

```text
4 passed
```

---

# Final Demo

A 5-minute end-to-end demonstration of the LearnSmart system will show:

1. GitHub repository
2. FastAPI Swagger documentation
3. Health endpoint
4. Personalized recommendation endpoint
5. Model Precision@5 result
6. Automated tests
7. Load-test results
8. Model card
9. GitHub Actions CI

### Demo Video



https://drive.google.com/file/d/1Ug_jl4Chg9fhZkWvNC3iqJzyRgK_KbMa/view?usp=sharing



---

# Repository

GitHub repository:

https://github.com/SnehaGowda18/LearnSmart-Recommendation

---

# Final Status

**LearnSmart Recommendation Engine — Completed**

The project provides an end-to-end hybrid course recommendation system with:

* Data pipeline
* Content-based recommendation
* Collaborative filtering
* Hybrid recommendation
* Model experimentation
* MLflow experiment tracking
* FastAPI REST API
* API integration testing
* Load testing
* Model card
* Bias/fairness audit
* Deployment documentation
* GitHub Actions CI
* Final end-to-end demonstration
