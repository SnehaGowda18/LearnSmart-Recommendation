import pandas as pd

from hybrid_model import recommend_courses


DATA_PATH = "data/learnsmart_interactions.csv"


def precision_at_5(df, user_id, content_weight, collaborative_weight):
    """Calculate Precision@5 for one user."""

    user_data = df[
        (df["user_id"] == user_id)
        & (df["completed"] == 1)
    ].copy()

    if len(user_data) < 2:
        return None

    # Hold out the last completed course.
    test_course = user_data.iloc[-1]["course_id"]

    # Remove the test course from the user's history.
    history = df[
        ~(
            (df["user_id"] == user_id)
            & (df["course_id"] == test_course)
        )
    ].copy()

    recommendations = recommend_courses(
        history,
        user_id,
        top_n=5,
        content_weight=content_weight,
        collaborative_weight=collaborative_weight,
    )

    recommended_courses = set(
        recommendations["course_id"]
    )

    if test_course in recommended_courses:
        return 1 / 5

    return 0.0


def evaluate_configuration(
    df,
    content_weight,
    collaborative_weight,
):
    """Evaluate one hybrid weight configuration."""

    scores = []

    for user_id in sorted(df["user_id"].unique()):
        score = precision_at_5(
            df,
            user_id,
            content_weight,
            collaborative_weight,
        )

        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)

    configurations = [
        (1.0, 0.0),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.0, 1.0),
    ]

    results = []

    print("LearnSmart Hybrid Weight Experiment")
    print("=" * 50)

    for content_weight, collaborative_weight in configurations:

        precision = evaluate_configuration(
            data,
            content_weight,
            collaborative_weight,
        )

        results.append(
            {
                "content_weight": content_weight,
                "collaborative_weight": collaborative_weight,
                "precision_at_5": precision,
            }
        )

        print(
            f"Content={content_weight:.1f} | "
            f"Collaborative={collaborative_weight:.1f} | "
            f"Precision@5={precision:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "precision_at_5",
        ascending=False,
    )

    results_df.to_csv(
        "results/hybrid_weight_experiment.csv",
        index=False,
    )

    print("\nExperiment Results")
    print("=" * 50)
    print(
        results_df.to_string(
            index=False
        )
    )

    best = results_df.iloc[0]

    print("\nBest configuration:")
    print(
        f"Content weight: {best['content_weight']:.1f}"
    )
    print(
        f"Collaborative weight: "
        f"{best['collaborative_weight']:.1f}"
    )
    print(
        f"Precision@5: "
        f"{best['precision_at_5']:.4f}"
    )

    print(
        "\nExperiment completed successfully."
    )