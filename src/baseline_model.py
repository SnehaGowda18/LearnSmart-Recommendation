import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/learnsmart_interactions.csv"


def load_data():
    """Load LearnSmart learner-course data."""
    return pd.read_csv(DATA_PATH)


def build_course_profiles(df):
    """Build TF-IDF profiles for courses."""

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

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    course_matrix = vectorizer.fit_transform(
        courses["profile"]
    )

    return courses, course_matrix


def build_sequence_scores(df, user_id, courses):
    """Score courses according to observed learning sequences."""

    user_history = df[
        (df["user_id"] == user_id)
        & (df["completed"] == 1)
    ]

    completed_courses = set(
        user_history["course_id"]
    )

    sequence_scores = {
        course_id: 0.0
        for course_id in courses["course_id"]
    }

    # Learn transitions from the sequence information
    # available in the training data.
    for completed_course in completed_courses:

        completed_rows = df[
            df["course_id"] == completed_course
        ]

        for _, row in completed_rows.iterrows():

            other_user = row["user_id"]
            current_sequence = row["sequence"]

            later_courses = df[
                (df["user_id"] == other_user)
                & (
                    df["sequence"]
                    > current_sequence
                )
            ]

            for _, later_row in later_courses.iterrows():

                next_course = later_row[
                    "course_id"
                ]

                if next_course not in completed_courses:

                    distance = (
                        later_row["sequence"]
                        - current_sequence
                    )

                    sequence_scores[
                        next_course
                    ] += 1.0 / distance

    max_score = max(
        sequence_scores.values(),
        default=0
    )

    if max_score > 0:
        sequence_scores = {
            course_id: score / max_score
            for course_id, score
            in sequence_scores.items()
        }

    return sequence_scores


def recommend_courses(df, user_id, top_n=5):
    """Generate sequence-aware recommendations."""

    courses, course_matrix = build_course_profiles(df)

    user_history = df[
        (df["user_id"] == user_id)
        & (df["completed"] == 1)
    ]

    completed_courses = set(
        user_history["course_id"]
    )

    # Cold-start recommendation.
    if not completed_courses:
        return (
            df.groupby(
                [
                    "course_id",
                    "course_name",
                    "category",
                    "level",
                ],
                as_index=False,
            )["rating"]
            .mean()
            .sort_values(
                "rating",
                ascending=False
            )
            .head(top_n)
        )

    completed_indices = courses[
        courses["course_id"].isin(
            completed_courses
        )
    ].index.tolist()

    completed_matrix = course_matrix[
        completed_indices
    ]

    user_profile = np.asarray(
        completed_matrix.mean(axis=0)
    )

    content_scores = cosine_similarity(
        user_profile,
        course_matrix
    ).flatten()

    recommendations = courses.copy()

    recommendations["content_score"] = (
        content_scores
    )

    sequence_scores = build_sequence_scores(
        df,
        user_id,
        courses
    )

    recommendations["sequence_score"] = (
        recommendations["course_id"]
        .map(sequence_scores)
        .fillna(0)
    )

    # Learner preference from completed courses.
    category_scores = (
        user_history.groupby("category")
        .apply(
            lambda x: (
                x["quiz_score"].mean() / 100
                + x["rating"].mean() / 5
            ) / 2,
            include_groups=False,
        )
        .to_dict()
    )

    recommendations["preference_score"] = (
        recommendations["category"]
        .map(category_scores)
        .fillna(0)
    )

    # Weighted recommendation score.
    recommendations["score"] = (
        0.20 * recommendations["content_score"]
        + 0.65 * recommendations["sequence_score"]
        + 0.15 * recommendations["preference_score"]
    )

    recommendations = recommendations[
        ~recommendations["course_id"].isin(
            completed_courses
        )
    ]

    recommendations = recommendations.sort_values(
        "score",
        ascending=False
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

    print(
        "LearnSmart Sequence-Aware Recommender"
    )
    print("=" * 50)

    user_id = "U001"

    recommendations = recommend_courses(
        data,
        user_id,
        top_n=5
    )

    print(f"\nRecommendations for {user_id}:")
    print(
        recommendations.to_string(
            index=False
        )
    )

    print(
        "\nRecommendation completed successfully."
    )