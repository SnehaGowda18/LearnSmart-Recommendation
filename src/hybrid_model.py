import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/learnsmart_interactions.csv"


def load_data():
    """Load LearnSmart interaction data."""
    return pd.read_csv(DATA_PATH)


def build_content_scores(df, user_id):
    """Calculate content-based similarity scores."""

    courses = (
        df[
            ["course_id", "course_name", "category", "level"]
        ]
        .drop_duplicates("course_id")
        .reset_index(drop=True)
    )

    courses["profile"] = (
        courses["course_name"].fillna("")
        + " "
        + courses["category"].fillna("")
        + " "
        + courses["level"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    course_matrix = vectorizer.fit_transform(courses["profile"])

    completed = set(
        df[
            (df["user_id"] == user_id)
            & (df["completed"] == 1)
        ]["course_id"]
    )

    if not completed:
        return pd.DataFrame(
            columns=["course_id", "content_score"]
        )

    completed_indices = courses[
        courses["course_id"].isin(completed)
    ].index.tolist()

    user_profile = course_matrix[
        completed_indices
    ].mean(axis=0)

    user_profile = user_profile.A

    similarity_scores = cosine_similarity(
        user_profile,
        course_matrix
    ).flatten()

    result = pd.DataFrame(
        {
            "course_id": courses["course_id"],
            "content_score": similarity_scores,
        }
    )

    return result[
        ~result["course_id"].isin(completed)
    ]


def build_collaborative_scores(df, user_id):
    """
    Calculate collaborative scores using learner
    course-history similarity.
    """

    user_history = df[
        (df["user_id"] == user_id)
        & (df["completed"] == 1)
    ]

    if user_history.empty:
        return pd.DataFrame(
            columns=[
                "course_id",
                "collaborative_score",
            ]
        )

    completed_courses = set(
        user_history["course_id"]
    )

    # Build a user-course interaction matrix.
    # Completed courses receive higher interaction values.
    data = df.copy()

    data["interaction"] = (
        data["completed"] * 1.0
        + (data["quiz_score"] / 100) * 0.3
    )

    matrix = data.pivot_table(
        index="user_id",
        columns="course_id",
        values="interaction",
        aggfunc="max",
        fill_value=0,
    )

    if user_id not in matrix.index:
        return pd.DataFrame(
            columns=[
                "course_id",
                "collaborative_score",
            ]
        )

    # Calculate similarity between learners.
    similarity_matrix = cosine_similarity(matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=matrix.index,
        columns=matrix.index,
    )

    similar_users = (
        similarity_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
    )

    scores = {}

    for similar_user, similarity in similar_users.items():

        if similarity <= 0:
            continue

        similar_history = df[
            df["user_id"] == similar_user
        ].sort_values("sequence")

        for _, row in similar_history.iterrows():

            course_id = row["course_id"]

            # Never recommend an already completed course.
            if course_id in completed_courses:
                continue

            # Ignore negative/unrelated courses.
            if row["sequence"] <= 0:
                continue

            # Earlier courses in a learner's path
            # receive stronger collaborative weight.
            sequence_weight = 1.0 / row["sequence"]

            score = (
                similarity
                * sequence_weight
            )

            scores[course_id] = (
                scores.get(course_id, 0)
                + score
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


def recommend_courses(
    df,
    user_id,
    top_n=5,
    content_weight=0.5,
    collaborative_weight=0.5,
):
    """
    Generate hybrid recommendations.

    Parameters
    ----------
    df : pandas.DataFrame
        LearnSmart interaction data.

    user_id : str
        Target learner.

    top_n : int
        Number of recommendations.

    content_weight : float
        Weight assigned to content-based model.

    collaborative_weight : float
        Weight assigned to collaborative model.
    """

    if content_weight < 0 or collaborative_weight < 0:
        raise ValueError(
            "Model weights cannot be negative."
        )

    total_weight = (
        content_weight
        + collaborative_weight
    )

    if total_weight == 0:
        raise ValueError(
            "At least one model weight must be greater than zero."
        )

    # Normalize weights automatically.
    content_weight = (
        content_weight / total_weight
    )

    collaborative_weight = (
        collaborative_weight / total_weight
    )

    content_scores = build_content_scores(
        df,
        user_id,
    )

    collaborative_scores = (
        build_collaborative_scores(
            df,
            user_id,
        )
    )

    # Course information.
    all_courses = (
        df[
            [
                "course_id",
                "course_name",
                "category",
                "level",
            ]
        ]
        .drop_duplicates("course_id")
        .reset_index(drop=True)
    )

    # Merge content scores.
    scores = all_courses.merge(
        content_scores,
        on="course_id",
        how="left",
    )

    # Merge collaborative scores.
    scores = scores.merge(
        collaborative_scores,
        on="course_id",
        how="left",
    )

    scores["content_score"] = (
        scores["content_score"]
        .fillna(0)
    )

    scores["collaborative_score"] = (
        scores["collaborative_score"]
        .fillna(0)
    )

    # Remove completed courses.
    completed_courses = set(
        df[
            (df["user_id"] == user_id)
            & (df["completed"] == 1)
        ]["course_id"]
    )

    scores = scores[
        ~scores["course_id"].isin(
            completed_courses
        )
    ].copy()

    # Normalize content scores.
    content_max = (
        scores["content_score"].max()
    )

    if content_max > 0:
        scores["content_normalized"] = (
            scores["content_score"]
            / content_max
        )
    else:
        scores["content_normalized"] = 0.0

    # Normalize collaborative scores.
    collaborative_max = (
        scores["collaborative_score"].max()
    )

    if collaborative_max > 0:
        scores["collaborative_normalized"] = (
            scores["collaborative_score"]
            / collaborative_max
        )
    else:
        scores["collaborative_normalized"] = 0.0

    # Final hybrid score.
    scores["hybrid_score"] = (
        content_weight
        * scores["content_normalized"]
        + collaborative_weight
        * scores["collaborative_normalized"]
    )

    # Stable sorting gives reproducible results.
    recommendations = (
        scores
        .sort_values(
            by=[
                "hybrid_score",
                "course_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(top_n)
    )

    return recommendations[
        [
            "course_id",
            "course_name",
            "category",
            "level",
            "content_normalized",
            "collaborative_normalized",
            "hybrid_score",
        ]
    ]


if __name__ == "__main__":

    data = load_data()

    print(
        "LearnSmart Hybrid Recommendation Engine"
    )
    print("=" * 50)

    user_id = "U001"

    recommendations = recommend_courses(
        data,
        user_id,
        top_n=5,
        content_weight=0.5,
        collaborative_weight=0.5,
    )

    print(
        f"\nHybrid Recommendations for {user_id}:"
    )

    print(
        recommendations.to_string(
            index=False
        )
    )

    print(
        "\nHybrid recommendation completed successfully."
    )