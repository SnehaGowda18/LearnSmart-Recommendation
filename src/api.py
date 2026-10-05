from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.hybrid_model import load_data, recommend_courses


app = FastAPI(
    title="LearnSmart Recommendation API",
    description="REST API for the LearnSmart hybrid course recommendation engine.",
    version="1.0.0",
)


# Load the dataset once when the API starts.
DATA = load_data()

VALID_USERS = set(DATA["user_id"].unique())


class RecommendationRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        description="Learner ID, for example U001",
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of courses to recommend",
    )


@app.get("/")
def root():
    """Return API information."""
    return {
        "message": "LearnSmart Recommendation API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    """Return API health and dataset information."""
    return {
        "status": "healthy",
        "users": int(DATA["user_id"].nunique()),
        "courses": int(DATA["course_id"].nunique()),
        "model": "hybrid",
        "content_weight": 0.3,
        "collaborative_weight": 0.7,
    }


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    """Generate personalized course recommendations."""
    if request.user_id not in VALID_USERS:
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.user_id}' not found.",
        )

    try:
        recommendations = recommend_courses(
            DATA,
            request.user_id,
            top_n=request.top_n,
            content_weight=0.3,
            collaborative_weight=0.7,
        )

        return {
            "user_id": request.user_id,
            "top_n": request.top_n,
            "model": "hybrid",
            "content_weight": 0.3,
            "collaborative_weight": 0.7,
            "recommendations": recommendations.to_dict(
                orient="records"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {exc}",
        ) from exc