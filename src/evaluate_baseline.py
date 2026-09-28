import pandas as pd

from baseline_model import recommend_courses

DATA_PATH = "data/learnsmart_interactions.csv"


def precision_at_5_for_user(df, user_id):
    """Calculate Precision@5 using known relevant future courses."""

    user_data = df[df["user_id"] == user_id]

    relevant_courses = set(
        user_data[
            (user_data["completed"] == 0)
            & (user_data["relevant"] == 1)
        ]["course_id"]
    )

    if not relevant_courses:
        return None

    recommendations = recommend_courses(
        df,
        user_id,
        top_n=5
    )

    recommended_courses = set(
        recommendations["course_id"]
    )

    hits = len(
        recommended_courses.intersection(
            relevant_courses
        )
    )

    return hits / 5


def evaluate_all_users(df):
    """Calculate average Precision@5 across all users."""

    scores = []

    for user_id in sorted(df["user_id"].unique()):
        score = precision_at_5_for_user(
            df,
            user_id
        )

        if score is not None:
            scores.append(score)

    if not scores:
        return 0.0, 0

    return sum(scores) / len(scores), len(scores)


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)

    precision, users_evaluated = evaluate_all_users(
        data
    )

    print("LearnSmart Baseline Evaluation")
    print("=" * 40)
    print(f"Users evaluated: {users_evaluated}")
    print(f"Precision@5: {precision:.4f}")
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
            "Target Precision@5 >= 0.60: NOT YET MET"
        )