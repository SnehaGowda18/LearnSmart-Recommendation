# LearnSmart Baseline Model Report

## Dataset

- Users: 100
- Courses: 20
- Total interactions: 900
- Completed interactions: 200
- Relevant future courses: 500

## Baseline Model

The baseline recommendation engine uses:

- TF-IDF content similarity
- Course sequence information
- Learner quiz scores
- Learner ratings
- Weighted recommendation scoring

## Evaluation Metric

### Precision@5

Precision@5 measures the proportion of the top 5 recommended courses that are relevant to the learner's future learning path.

```text
Precision@5 = Relevant recommendations in Top-5 / 5


So it should look exactly like:

```markdown
Precision@5 = Relevant recommendations in Top-5 / 5

## Result

Users evaluated: 100

Precision@5: 0.9000

Precision@5: 90.00%

## Acceptance Criteria

Required:

Precision@5 >= 0.60

Achieved:

Precision@5 = 0.90

Status:

**PASSED**

## Conclusion

The sequence-aware baseline recommender achieved the required Precision@5 threshold on the LearnSmart evaluation dataset. The model recommends relevant future courses based on the learner's completed courses and learning sequence.