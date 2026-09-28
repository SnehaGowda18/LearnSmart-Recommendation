# LearnSmart EdTech Recommendation System

## 1. System Architecture

The system will use a hybrid recommendation approach combining Collaborative Filtering and Content-Based Filtering.

```text
                    LearnSmart User
                           |
                           v
              +-------------------------+
              | Learner Interaction Data |
              | Completed Courses       |
              | Quiz Scores              |
              +------------+------------+
                           |
                           v
                  Data Preprocessing
                           |
              +------------+------------+
              |                         |
              v                         v
      Collaborative Model       Content-Based Model
              |                         |
              |                         |
              +------------+------------+
                           |
                           v
                 Hybrid Recommendation
                           |
                           v
                    Course Ranking
                           |
                           v
                    Top-5 Courses
                           |
                           v
                    REST API Output
```

## 2. Data Inputs

The recommendation system will use:

* Learner IDs
* Course IDs
* Completed courses
* Quiz scores
* Course information/features

The exact available fields will be confirmed during the data-ingestion milestone after inspecting the provided datasets.

## 3. Collaborative Filtering

Collaborative Filtering will use learner-course interaction information.

Example:

```text
Learner A → Python → Completed
Learner A → SQL → Completed
Learner B → Python → Completed
Learner B → SQL → Completed
Learner B → Machine Learning → Completed
```

Learners with similar course interaction patterns can help identify additional courses that may be relevant.

## 4. Content-Based Filtering

Content-Based Filtering will use course characteristics and the learner's previous course history.

The model will compare the learner's profile with available course features and generate similarity-based recommendations.

## 5. Hybrid Recommendation

The two recommendation components will be combined:

```text
Hybrid Score =
    Collaborative Score × CF Weight
    +
    Content Score × Content Weight
```

The weights will be treated as configurable parameters and evaluated during experimentation.

## 6. Warm-Start Recommendation

For learners with existing course history:

```text
Completed Courses
        ↓
Learner Profile
        ↓
Collaborative + Content-Based Models
        ↓
Hybrid Scores
        ↓
Top-5 Recommendations
```

## 7. Cold-Start Recommendation

For learners with insufficient historical interaction data:

```text
New Learner
     ↓
Available learner/course information
     ↓
Content-Based / fallback recommendation
     ↓
Course Ranking
     ↓
Top-5 Recommendations
```

The exact cold-start strategy will be finalized after inspecting the available training data.

## 8. Evaluation

Primary acceptance metric:

**Precision@5 >= 0.60**

Additional metrics:

* Recall@5
* F1@5
* Coverage
* Latency

## 9. REST API

The recommendation engine will later be exposed through FastAPI.

Example endpoint:

```text
POST /recommend
```

Example request:

```json
{
  "user_id": 101
}
```

Example response:

```json
{
  "user_id": 101,
  "recommendations": [
    12,
    25,
    31,
    44,
    50
  ]
}
```

The final API structure will be implemented during the integration milestone.

## 10. A/B Testing

The project will include an A/B test using 100 sample users.

The experiment will compare two recommendation configurations and measure recommendation performance using predefined evaluation metrics.

## 11. Project Flow

```text
Data
  ↓
Preprocessing
  ↓
Baseline Model
  ↓
Collaborative Filtering
  +
Content-Based Filtering
  ↓
Hybrid Model
  ↓
Evaluation
  ↓
Experiment Tracking
  ↓
FastAPI Integration
  ↓
Testing + Load Testing
  ↓
A/B Testing
  ↓
Final Demo + Model Card
```
