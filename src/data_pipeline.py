import pandas as pd


DATA_PATH = "data/learnsmart_interactions.csv"


def load_data():
    """Load the LearnSmart interaction dataset."""
    return pd.read_csv(DATA_PATH)


def preprocess_data(df):
    """Clean and prepare learner-course interaction data."""
    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Validate required columns
    required_columns = [
        "user_id",
        "course_id",
        "course_name",
        "category",
        "level",
        "completed",
        "quiz_score",
        "rating",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Convert numeric columns
    df["completed"] = pd.to_numeric(df["completed"], errors="coerce")
    df["quiz_score"] = pd.to_numeric(df["quiz_score"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Remove rows with invalid essential values
    df = df.dropna(
        subset=[
            "user_id",
            "course_id",
            "completed",
            "quiz_score",
            "rating",
        ]
    )

    # Keep valid ranges
    df = df[
        df["completed"].isin([0, 1])
        & df["quiz_score"].between(0, 100)
        & df["rating"].between(0, 5)
    ]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = load_data()
    clean_data = preprocess_data(data)

    print("Original shape:", data.shape)
    print("Cleaned shape:", clean_data.shape)
    print("Users:", clean_data["user_id"].nunique())
    print("Courses:", clean_data["course_id"].nunique())
    print("\nMissing values:")
    print(clean_data.isnull().sum())
    print("\nPreprocessing completed successfully.")