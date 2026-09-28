import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/learnsmart_interactions.csv"


def load_data():
    """Load LearnSmart interaction data."""
    return pd.read_csv(DATA_PATH)


def build_user_item_matrix(df):
    """Create a user-course interaction matrix."""

    # Combine completion and quiz performance into an interaction score.
    data = df.copy()

    data["interaction_score"] = (
        data["completed"] * 0.7
        + (data["quiz_score"] / 100) * 0.3
    )

    user_item_matrix = data.pivot_table(
        index="user_id",
        columns="course_id",
        values="interaction_score",
        aggfunc="mean",
        fill_value=0,
    )

    return user_item_matrix


def recommend_courses(df, user_id, top_n=5):
    """Generate collaborative-filtering recommendations."""

    matrix = build_user_item_matrix(df)

    if user_id not in matrix.index:
        return pd.DataFrame(
            columns=[
                "course_id",
                "course_name",
                "category",
                "level",
                "score",
            ]
        )

    # Calculate similarity between users.
    user_similarity = cosine_similarity(matrix)

    similarity_df = pd.DataFrame(
        user_similarity,
        index=matrix.index,
        columns=matrix.index,
    )

    similar_users = (
        similarity_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
    )

    # Courses already completed by the target user.
    completed_courses = set(
        df[
            (df["user_id"] == user_id)
            & (df["completed"] == 1)
        ]["course_id"]
    )

    scores = {}

    # Use similar users to generate course scores.
    for similar_user, similarity in similar_users.items():

        if similarity <= 0:
            continue

        similar_user_courses = matrix.loc[similar_user]

        for course_id, interaction in similar_user_courses.items():

            if course_id in completed_courses:
                continue

            if interaction <= 0:
                continue

            scores[course_id] = (
                scores.get(course_id, 0)
                + similarity * interaction
            )

    if not scores:
        return pd.DataFrame(
            columns=[
                "course_id",
                "course_name",
                "category",
                "level",
                "score",
            ]
        )

    recommendations = pd.DataFrame(
        [
            {
                "course_id": course_id,
                "score": score,
            }
            for course_id, score in scores.items()
        ]
    )

    course_details = df[
        [
            "course_id",
            "course_name",
            "category",
            "level",
        ]
    ].drop_duplicates("course_id")

    recommendations = recommendations.merge(
        course_details,
        on="course_id",
        how="left",
    )

    recommendations = recommendations.sort_values(
        "score",
        ascending=False,
    ).head(top_n)

    return recommendations[
        [
            "course_id",
            "course_name",
            "category",
            "level",
            "score",
        ]
    ]


if __name__ == "__main__":
    data = load_data()

    print("LearnSmart Collaborative Filtering Recommender")
    print("=" * 50)

    user_id = "U001"

    recommendations = recommend_courses(
        data,
        user_id,
        top_n=5,
    )

    print(f"\nRecommendations for {user_id}:")

    if recommendations.empty:
        print("No collaborative recommendations available.")
    else:
        print(
            recommendations.to_string(
                index=False
            )
        )

    print(
        "\nCollaborative filtering completed successfully."
    )