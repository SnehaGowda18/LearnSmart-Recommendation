import pandas as pd

from improved_model import recommend_courses


DATA_PATH = "data/learnsmart_interactions.csv"


def precision_at_5_for_user(df, user_id):
    """Evaluate one held-out completed course."""

    user_data = df[
        (df["user_id"] == user_id)
        & (df["completed"] == 1)
    ].copy()

    if len(user_data) < 2:
        return None

    # Hold out the last completed course.
    test_course = user_data.iloc[-1]["course_id"]

    # Remove the test course from the learner's history.
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
    )

    recommended_courses = set(
        recommendations["course_id"]
    )

    if test_course in recommended_courses:
        return 1 / 5

    return 0.0


def evaluate_all_users(df):
    """Calculate average Precision@5."""

    scores = []

    for user_id in sorted(df["user_id"].unique()):

        score = precision_at_5_for_user(
            df,
            user_id,
        )

        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0, 0

    return sum(scores) / len(scores), len(scores)


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)

    precision, users_evaluated = (
        evaluate_all_users(data)
    )

    print("LearnSmart Improved Model Evaluation")
    print("=" * 50)
    print(
        f"Users evaluated: {users_evaluated}"
    )
    print(
        f"Precision@5: {precision:.4f}"
    )
    print(
        f"Precision@5 (%): "
        f"{precision * 100:.2f}%"
    )

    if precision >= 0.60:
        print(
            "Target Precision@5 >= 0.60: PASSED"
        )
    else:
        print(
            "Target Precision@5 >= 0.60: "
            "NOT YET MET"
        )