import os

import mlflow
import pandas as pd

from hybrid_model import recommend_courses


DATA_PATH = "data/learnsmart_interactions.csv"
RESULTS_PATH = "results/hyperparameter_results.csv"

EXPERIMENT_NAME = "LearnSmart_Hybrid_Recommender"


CONFIGURATIONS = [
    {
        "name": "config_A",
        "content_weight": 0.3,
        "collaborative_weight": 0.7,
    },
    {
        "name": "config_B",
        "content_weight": 0.7,
        "collaborative_weight": 0.3,
    },
]


def precision_at_5_for_user(
    df,
    user_id,
    content_weight,
    collaborative_weight,
):
    """Calculate Precision@5 for one learner."""

    user_data = df[
        df["user_id"] == user_id
    ]

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
        top_n=5,
        content_weight=content_weight,
        collaborative_weight=collaborative_weight,
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


def evaluate_configuration(
    df,
    content_weight,
    collaborative_weight,
):
    """Evaluate one hybrid-model configuration."""

    scores = []

    for user_id in sorted(
        df["user_id"].unique()
    ):
        score = precision_at_5_for_user(
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


def main():
    """Run and track all hyperparameter experiments."""

    data = pd.read_csv(DATA_PATH)

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    results = []

    print(
        "LearnSmart MLflow Hyperparameter Experiments"
    )
    print("=" * 60)

    for config in CONFIGURATIONS:

        precision = evaluate_configuration(
            data,
            config["content_weight"],
            config["collaborative_weight"],
        )

        with mlflow.start_run(
            run_name=config["name"]
        ):

            # Log hyperparameters.
            mlflow.log_param(
                "content_weight",
                config["content_weight"],
            )

            mlflow.log_param(
                "collaborative_weight",
                config["collaborative_weight"],
            )

            mlflow.log_param(
                "top_n",
                5,
            )

            # Log dataset information.
            mlflow.log_param(
                "dataset_rows",
                len(data),
            )

            mlflow.log_param(
                "users",
                data["user_id"].nunique(),
            )

            mlflow.log_param(
                "courses",
                data["course_id"].nunique(),
            )

            # Log evaluation metric.
            mlflow.log_metric(
                "precision_at_5",
                precision,
            )

            results.append(
                {
                    "configuration": config["name"],
                    "content_weight": config[
                        "content_weight"
                    ],
                    "collaborative_weight": config[
                        "collaborative_weight"
                    ],
                    "precision_at_5": precision,
                }
            )

            print(
                f"\n{config['name']}"
            )

            print(
                f"Content weight: "
                f"{config['content_weight']}"
            )

            print(
                f"Collaborative weight: "
                f"{config['collaborative_weight']}"
            )

            print(
                f"Precision@5: "
                f"{precision:.4f}"
            )

    results_df = pd.DataFrame(results)

    os.makedirs(
        "results",
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    best_result = results_df.loc[
        results_df["precision_at_5"].idxmax()
    ]

    print("\n" + "=" * 60)
    print("Best Configuration")
    print("=" * 60)

    print(
        f"Configuration: "
        f"{best_result['configuration']}"
    )

    print(
        f"Content weight: "
        f"{best_result['content_weight']}"
    )

    print(
        f"Collaborative weight: "
        f"{best_result['collaborative_weight']}"
    )

    print(
        f"Precision@5: "
        f"{best_result['precision_at_5']:.4f}"
    )

    print(
        f"\nResults saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        "\nMLflow experiments completed successfully."
    )


if __name__ == "__main__":
    main()