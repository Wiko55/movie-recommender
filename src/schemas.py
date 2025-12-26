from typing import List

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[str]
    model_version: str


class HealthCheck(BaseModel):
    status: str
    running_model: bool


class LandingPage(BaseModel):
    status_serwera: str
    messege: str
