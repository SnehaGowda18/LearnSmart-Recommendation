# LearnSmart EdTech Recommendation System — Research

## 1. Research Objective

The LearnSmart project requires an adaptive learning recommendation engine that recommends suitable next courses using:

* Collaborative Filtering
* Content-Based Filtering
* Quiz scores
* Completed courses
* Hybrid recommendation techniques

The system must support both warm-start and cold-start users and be evaluated using Precision@5.

## 2. Reference 1 — Recommender Systems Survey

**Paper:** Revisiting Recommender Systems: An Investigative Survey
**Published:** 2025

This survey reviews major recommender-system approaches, including collaborative filtering, content-based recommendation, and hybrid systems. It also discusses common challenges such as the cold-start problem.

**Relevance to LearnSmart:**

* Helps identify suitable recommendation approaches.
* Highlights cold-start as an important design problem.
* Supports using a hybrid recommendation strategy.

## 3. Reference 2 — Content-Based Recommendation

**Paper:** Content-based Recommender Systems: State of the Art and Trends
**Authors:** Pasquale Lops, Marco de Gemmis, Giovanni Semeraro

Content-based recommendation builds user profiles from preferences and matches them with item attributes to recommend similar items.

**Relevance to LearnSmart:**

* Course attributes can be used to represent learning content.
* Completed courses can help create a learner preference profile.
* Useful for recommending courses to new users when collaborative data is limited.

## 4. Reference 3 — Hybrid Recommendation

**Paper:** Combining Content-Based and Collaborative Recommendations: A Hybrid Approach Based on Bayesian Networks
**Published:** 2010

The paper combines collaborative and content-based recommendation methods to address limitations of using either approach independently.

**Relevance to LearnSmart:**

* Supports combining learner interaction data with course-content information.
* Provides a research basis for a hybrid recommendation engine.
* Hybrid recommendation is suitable for handling different types of learner information.

## 5. Research Findings

Based on the reviewed references:

1. Collaborative Filtering can use interactions between learners and courses.
2. Content-Based Filtering can use course characteristics and learner preferences.
3. Hybrid recommendation combines information from both approaches.
4. Cold-start needs to be explicitly handled.
5. Ranking metrics such as Precision@5 are appropriate for evaluating top-course recommendations.

## 6. Proposed Direction for LearnSmart

The initial system will use a hybrid architecture:

**Learner Data**
→ completed courses + quiz scores

**Course Data**
→ course features/content

**Collaborative Filtering**
→ learner-course interaction patterns

**Content-Based Filtering**
→ similarity between learner interests and course features

**Hybrid Recommendation**
→ combine both recommendation scores

**Ranking**
→ rank candidate courses

**Top-K Output**
→ recommend the next 5 courses

## 7. Initial Evaluation Metrics

Primary metric:

* Precision@5

Additional metrics to consider:

* Recall@5
* F1@5
* Coverage
* Recommendation latency

The acceptance target for the project is:

**Precision@5 >= 0.60**

## 8. Conclusion

The research supports a hybrid recommendation approach for LearnSmart. Collaborative Filtering can capture patterns from learner-course interactions, while Content-Based Filtering can use course characteristics and learner preferences. Combining both approaches provides the foundation for handling warm-start and cold-start recommendation scenarios.
