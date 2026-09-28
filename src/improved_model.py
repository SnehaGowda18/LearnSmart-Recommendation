import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/learnsmart_interactions.csv"


def load_data():
    """Load LearnSmart interaction data."""
    return pd.read_csv(DATA_PATH)


def min_max_normalize(values):
    """Normalize values to the 0-1 range."""
    values = np.asarray(values, dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def build_content_scores(df, user_id):
    """Build content scores using learner history and course metadata."""

    courses = df[
        ["course_id", "course_name", "category", "level"]
    ].drop_duplicates().reset_index(drop=True)

    courses["profile"] = (
        courses["course_name"].fillna("")
        + " "
        + courses["category"].fillna("")
        + " "
        + courses["level"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english")

    course_matrix = vectorizer.fit_transform(
        courses["profile"]
    )

    user_history = df[
        (df["user_id"] == user_id)
        & (df["completed"] == 1)
    ].copy()

    completed_courses = set(
        user_history["course_id"]
    )

    if not completed_courses:
        return pd.DataFrame(
            columns=[
                "course_id",
                "content_score",
            ]
        )

    history_indices = courses[
        courses["course_id"].isin(
            completed_courses
        )
    ].index.tolist()

    # Give more importance to courses where
    # the learner performed well.
    weights = (
        user_history["quiz_score"].values / 100
        + user_history["rating"].values / 5
    ) / 2

    weights = np.asarray(
        weights,
        dtype=float
    )

    history_matrix = course_matrix[
        history_indices
    ]

    weighted_profile = (
        history_matrix.multiply(
            weights.reshape(-1, 1)
        ).sum(axis=0)
    )

    weighted_profile = np.asarray(
        weighted_profile
    )

    scores = cosine_similarity(
        weighted_profile,
        course_matrix
    ).flatten()

    result = pd.DataFrame({
        "course_id": courses["course_id"],
        "content_score": scores,
    })

    return result[
        ~result["course_id"].isin(
            completed_courses
        )
    ]


def build_collaborative_scores(df, user_id):
    """Build collaborative scores using learner interactions."""

    data = df.copy()

    data["interaction_score"] = (
        data["completed"] * 0.6
        + (data["quiz_score"] / 100) * 0.25
        + (data["rating"] / 5) * 0.15
    )

    matrix = data.pivot_table(
        index="user_id",
        columns="course_id",
        values="interaction_score",
        aggfunc="mean",
        fill_value=0,
    )

    if user_id not in matrix.index:
        return pd.DataFrame(
            columns=[
                "course_id",
                "collaborative_score",
            ]
        )

    similarities = cosine_similarity(matrix)

    similarity_df = pd.DataFrame(
        similarities,
        index=matrix.index,
        columns=matrix.index,
    )

    similar_users = (
        similarity_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
    )

    completed_courses = set(
        df[
            (df["user_id"] == user_id)
            & (df["completed"] == 1)
        ]["course_id"]
    )

    scores = {}

    for similar_user, similarity in similar_users.items():

        if similarity <= 0:
            continue

        user_courses = matrix.loc[
            similar_user
        ]

        for course_id, interaction in user_courses.items():

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
                "collaborative_score",
            ]
        )

    result = pd.DataFrame(
        [
            {
                "course_id": course_id,
                "collaborative_score": score,
            }
            for course_id, score in scores.items()
        ]
    )

    return result


def build_learner_preference_scores(
    df,
    user_id,
):
    """Score courses according to learner preferences."""

    user_history = df[
        (df["user_id"] == user_id)
        & (df["completed"] == 1)
    ].copy()

    courses = df[
        [
            "course_id",
            "course_name",
            "category",
            "level",
        ]
    ].drop_duplicates("course_id")

    if user_history.empty:
        courses["preference_score"] = 0.0
        return courses[
            [
                "course_id",
                "preference_score",
            ]
        ]

    # Calculate category preference.
    category_scores = (
        user_history.groupby("category")[
            "quiz_score"
        ].mean()
    )

    category_rating = (
        user_history.groupby("category")[
            "rating"
        ].mean()
        / 5
    )

    preference = (
        category_scores / 100 * 0.7
        + category_rating * 0.3
    )

    courses["preference_score"] = (
        courses["category"]
        .map(preference)
        .fillna(0)
    )

    return courses[
        [
            "course_id",
            "preference_score",
        ]
    ]


def recommend_courses(
    df,
    user_id,
    top_n=5,
    content_weight=0.25,
    collaborative_weight=0.50,
    preference_weight=0.25,
):
    """Generate improved hybrid recommendations."""

    all_courses = df[
        [
            "course_id",
            "course_name",
            "category",
            "level",
        ]
    ].drop_duplicates("course_id")

    content = build_content_scores(
        df,
        user_id,
    )

    collaborative = build_collaborative_scores(
        df,
        user_id,
    )

    preference = build_learner_preference_scores(
        df,
        user_id,
    )

    result = all_courses.merge(
        content,
        on="course_id",
        how="left",
    )

    result = result.merge(
        collaborative,
        on="course_id",
        how="left",
    )

    result = result.merge(
        preference,
        on="course_id",
        how="left",
    )

    result = result.fillna(0)

    completed_courses = set(
        df[
            (df["user_id"] == user_id)
            & (df["completed"] == 1)
        ]["course_id"]
    )

    result = result[
        ~result["course_id"].isin(
            completed_courses
        )
    ]

    result["content_normalized"] = (
        min_max_normalize(
            result["content_score"]
        )
    )

    result["collaborative_normalized"] = (
        min_max_normalize(
            result["collaborative_score"]
        )
    )

    result["preference_normalized"] = (
        min_max_normalize(
            result["preference_score"]
        )
    )

    result["final_score"] = (
        content_weight
        * result["content_normalized"]
        + collaborative_weight
        * result["collaborative_normalized"]
        + preference_weight
        * result["preference_normalized"]
    )

    return result.sort_values(
        "final_score",
        ascending=False,
    ).head(top_n)[
        [
            "course_id",
            "course_name",
            "category",
            "level",
            "content_normalized",
            "collaborative_normalized",
            "preference_normalized",
            "final_score",
        ]
    ]


if __name__ == "__main__":
    data = load_data()

    print(
        "LearnSmart Improved Recommendation Engine"
    )
    print("=" * 55)

    user_id = "U001"

    recommendations = recommend_courses(
        data,
        user_id,
        top_n=5,
    )

    print(
        f"\nRecommendations for {user_id}:"
    )

    print(
        recommendations.to_string(
            index=False
        )
    )

    print(
        "\nImproved recommendation completed successfully."
    )