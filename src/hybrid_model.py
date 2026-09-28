import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/learnsmart_interactions.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def build_content_scores(df, user_id):
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

    indices = courses[
        courses["course_id"].isin(completed)
    ].index.tolist()

    user_profile = course_matrix[indices].mean(axis=0)

    user_profile = user_profile.A

    scores = cosine_similarity(
        user_profile,
        course_matrix
    ).flatten()

    result = pd.DataFrame({
        "course_id": courses["course_id"],
        "content_score": scores,
    })

    return result[
        ~result["course_id"].isin(completed)
    ]


def build_collaborative_scores(df, user_id):
    data = df.copy()

    data["interaction_score"] = (
        data["completed"] * 0.7
        + (data["quiz_score"] / 100) * 0.3
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
            columns=["course_id", "collaborative_score"]
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

    completed = set(
        df[
            (df["user_id"] == user_id)
            & (df["completed"] == 1)
        ]["course_id"]
    )

    scores = {}

    for similar_user, similarity in similar_users.items():

        if similarity <= 0:
            continue

        for course_id, interaction in (
            matrix.loc[similar_user].items()
        ):

            if course_id in completed:
                continue

            if interaction <= 0:
                continue

            scores[course_id] = (
                scores.get(course_id, 0)
                + similarity * interaction
            )

    if not scores:
        return pd.DataFrame(
            columns=["course_id", "collaborative_score"]
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
    """Generate hybrid recommendations."""

    content_scores = build_content_scores(
        df,
        user_id
    )

    collaborative_scores = build_collaborative_scores(
        df,
        user_id
    )

    all_courses = df[
        ["course_id", "course_name", "category", "level"]
    ].drop_duplicates("course_id")

    scores = all_courses.merge(
        content_scores,
        on="course_id",
        how="left",
    )

    scores = scores.merge(
        collaborative_scores,
        on="course_id",
        how="left",
    )

    scores["content_score"] = scores[
        "content_score"
    ].fillna(0)

    scores["collaborative_score"] = scores[
        "collaborative_score"
    ].fillna(0)

    completed = set(
        df[
            (df["user_id"] == user_id)
            & (df["completed"] == 1)
        ]["course_id"]
    )

    scores = scores[
        ~scores["course_id"].isin(completed)
    ]

    # Normalize both models to the same 0-1 scale.
    if scores["content_score"].max() > 0:
        scores["content_normalized"] = (
            scores["content_score"]
            / scores["content_score"].max()
        )
    else:
        scores["content_normalized"] = 0

    if scores["collaborative_score"].max() > 0:
        scores["collaborative_normalized"] = (
            scores["collaborative_score"]
            / scores["collaborative_score"].max()
        )
    else:
        scores["collaborative_normalized"] = 0

    scores["hybrid_score"] = (
        content_weight
        * scores["content_normalized"]
        + collaborative_weight
        * scores["collaborative_normalized"]
    )

    return scores.sort_values(
        "hybrid_score",
        ascending=False
    ).head(top_n)[
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

    print("LearnSmart Hybrid Recommendation Engine")
    print("=" * 50)

    user_id = "U001"

    recommendations = recommend_courses(
        data,
        user_id,
        top_n=5,
        content_weight=0.5,
        collaborative_weight=0.5,
    )

    print(f"\nHybrid Recommendations for {user_id}:")
    print(
        recommendations.to_string(
            index=False
        )
    )

    print(
        "\nHybrid recommendation completed successfully."
    )