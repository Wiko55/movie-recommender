from typing import List

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[str]
    source: str
    model_version: str


class HealthCheck(BaseModel):
    status: str
    running_model: bool


class LandingPage(BaseModel):
    status_serwera: str
    messege: str


# ------------------------------#
class MovieBase(BaseModel):
    title: str
    genres: str | None = None


class Movie(MovieBase):
    id: int

    class Config:
        from_atributes = True


class RatingCreate(BaseModel):
    user_id: int
    movie_id: int
    rating: float
