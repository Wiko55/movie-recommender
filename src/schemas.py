from typing import List

from pydantic import BaseModel, ConfigDict


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


class MovieBase(BaseModel):
    title: str
    genres: str | None = None


class Movie(MovieBase):
    id: int
    model_config = ConfigDict(from_attributes=True)


class RatingCreate(BaseModel):
    user_id: int
    movie_id: int
    rating: float
