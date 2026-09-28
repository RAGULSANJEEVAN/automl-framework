from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.recommendation_service import RecommendationService

router = APIRouter()

service = RecommendationService()


class RecommendationRequest(BaseModel):

    dataset_id: str


@router.post("/recommend")
def recommend(request: RecommendationRequest):

    try:

        result = service.recommend(
            request.dataset_id
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )