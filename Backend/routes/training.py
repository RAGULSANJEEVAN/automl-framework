from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.training_service import TrainingService

router = APIRouter()

service = TrainingService()


class TrainingRequest(BaseModel):
    dataset_id: str


@router.post("/train")
def train_model(request: TrainingRequest):

    try:

        result = service.train(request.dataset_id)

        return {
            "success": True,
            "message": "Model training completed successfully.",
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )