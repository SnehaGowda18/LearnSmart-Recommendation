# LearnSmart Checkpoint 3 - Core Model Experimentation

## 1. Objective

The objective of Checkpoint 3 was to train and evaluate the primary hybrid recommendation model using multiple hyperparameter configurations and track the experiments using MLflow.

## 2. Dataset

- Total interactions: 900
- Users: 100
- Courses: 20
- Completed interactions: 200
- Relevant future courses: 500

## 3. Primary Model

The LearnSmart recommendation engine uses a hybrid recommendation approach combining:

- Content-based filtering using TF-IDF and cosine similarity
- Collaborative filtering using learner course-history similarity
- Hybrid weighted scoring

The model supports configurable content and collaborative weights.

## 4. Hyperparameter Configurations

Two configurations were evaluated.

| Configuration | Content Weight | Collaborative Weight |
|---|---:|---:|
| Config A | 0.3 | 0.7 |
| Config B | 0.7 | 0.3 |

## 5. Evaluation Metric

### Precision@5

Precision@5 measures the proportion of relevant courses present in the top five recommendations.

The evaluation was performed across 100 learners using the known relevant future courses in the LearnSmart evaluation dataset.

## 6. Experiment Results

| Configuration | Content Weight | Collaborative Weight | Precision@5 |
|---|---:|---:|---:|
| Config A | 0.3 | 0.7 | 0.9440 (94.40%) |
| Config B | 0.7 | 0.3 | 0.6980 (69.80%) |

## 7. Best Configuration

The best-performing configuration was:

- Configuration: Config A
- Content weight: 0.3
- Collaborative weight: 0.7
- Precision@5: 0.9440 (94.40%)

The required target was Precision@5 >= 0.60 (60%).

Therefore, the best configuration exceeded the target.

## 8. MLflow Tracking

MLflow experiment name:

`LearnSmart_Hybrid_Recommender`

Two MLflow runs were successfully completed:

- `config_A` - FINISHED
- `config_B` - FINISHED

MLflow tracked:

- Content weight
- Collaborative weight
- Top-N value
- Dataset size
- Number of users
- Number of courses
- Precision@5

## 9. Evidence

The experiment results are also stored in:

`results/hyperparameter_results.csv`

MLflow tracking data is stored locally in the project's `mlruns` directory.

## 10. Conclusion

The hybrid recommendation model was evaluated using two different hyperparameter configurations. Config A, using a 0.3 content weight and 0.7 collaborative weight, achieved the best Precision@5 of 0.9440 (94.40%).

The experiment demonstrates that the collaborative component contributed strongly to recommendation quality on the LearnSmart evaluation dataset.

The best configuration will be used as the primary configuration for the next integration and serving stage.