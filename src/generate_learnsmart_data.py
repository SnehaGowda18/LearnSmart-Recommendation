import random
import pandas as pd

random.seed(42)

OUTPUT_PATH = "data/learnsmart_interactions.csv"

COURSES = [
    ("C001", "Python Programming", "Programming", "Beginner"),
    ("C002", "Data Structures and Algorithms", "Programming", "Intermediate"),
    ("C003", "Machine Learning Basics", "AI/ML", "Beginner"),
    ("C004", "Deep Learning Fundamentals", "AI/ML", "Intermediate"),
    ("C005", "SQL Fundamentals", "Data", "Beginner"),
    ("C006", "Data Analysis with Python", "Data", "Intermediate"),
    ("C007", "Cybersecurity Fundamentals", "Cybersecurity", "Beginner"),
    ("C008", "Ethical Hacking", "Cybersecurity", "Intermediate"),
    ("C009", "Cloud Computing Basics", "Cloud", "Beginner"),
    ("C010", "AWS Cloud Practitioner", "Cloud", "Intermediate"),
    ("C011", "Web Development with HTML CSS", "Web Development", "Beginner"),
    ("C012", "Backend Development with FastAPI", "Web Development", "Intermediate"),
    ("C013", "Statistics for Data Science", "Data", "Beginner"),
    ("C014", "Natural Language Processing", "AI/ML", "Intermediate"),
    ("C015", "DevOps Fundamentals", "DevOps", "Beginner"),
    ("C016", "Docker and Kubernetes", "DevOps", "Intermediate"),
    ("C017", "Computer Networks", "Networking", "Beginner"),
    ("C018", "Network Security", "Networking", "Intermediate"),
    ("C019", "Generative AI Fundamentals", "AI/ML", "Beginner"),
    ("C020", "MLOps with MLflow", "AI/ML", "Advanced"),
]

LEARNING_PATHS = [
    ["C001", "C002", "C003", "C004", "C014", "C019", "C020"],
    ["C013", "C005", "C006", "C003", "C004", "C014", "C020"],
    ["C017", "C018", "C007", "C008", "C009", "C010", "C016"],
    ["C009", "C010", "C015", "C016", "C012", "C003", "C020"],
    ["C011", "C012", "C015", "C016", "C009", "C010", "C020"],
    ["C003", "C004", "C014", "C019", "C020", "C006", "C005"],
]


def create_dataset():
    """Create sequential learner-course interactions."""

    course_lookup = {
        course_id: {
            "course_name": course_name,
            "category": category,
            "level": level,
        }
        for course_id, course_name, category, level in COURSES
    }

    rows = []

    for user_number in range(1, 101):
        user_id = f"U{user_number:03d}"

        path = LEARNING_PATHS[
            (user_number - 1) % len(LEARNING_PATHS)
        ]

        for position, course_id in enumerate(
            path,
            start=1
        ):
            course = course_lookup[course_id]

            is_completed = position <= 2
            is_relevant = position >= 3

            if is_completed:
                quiz_score = random.randint(
                    82,
                    95
                )
                rating = round(
                    random.uniform(4.0, 5.0),
                    1
                )
            else:
                quiz_score = 0
                rating = 0

            rows.append(
                {
                    "user_id": user_id,
                    "course_id": course_id,
                    "course_name": course["course_name"],
                    "category": course["category"],
                    "level": course["level"],
                    "sequence": position,
                    "completed": int(is_completed),
                    "quiz_score": quiz_score,
                    "rating": rating,
                    "relevant": int(is_relevant),
                }
            )

        used_courses = set(path)

        negative_candidates = [
            course_id
            for course_id, _, _, _
            in COURSES
            if course_id not in used_courses
        ]

        negative_courses = random.sample(
            negative_candidates,
            2
        )

        for course_id in negative_courses:
            course = course_lookup[course_id]

            rows.append(
                {
                    "user_id": user_id,
                    "course_id": course_id,
                    "course_name": course["course_name"],
                    "category": course["category"],
                    "level": course["level"],
                    "sequence": 0,
                    "completed": 0,
                    "quiz_score": 0,
                    "rating": 0,
                    "relevant": 0,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = create_dataset()

    data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("LearnSmart synthetic dataset created.")
    print("=" * 45)
    print(f"Rows: {len(data)}")
    print(f"Users: {data['user_id'].nunique()}")
    print(f"Courses: {data['course_id'].nunique()}")
    print(
        f"Completed interactions: "
        f"{data['completed'].sum()}"
    )
    print(
        f"Relevant future courses: "
        f"{data['relevant'].sum()}"
    )
    print(
        f"Sequence range: "
        f"{data['sequence'].min()}-"
        f"{data['sequence'].max()}"
    )
    print(f"Saved to: {OUTPUT_PATH}")