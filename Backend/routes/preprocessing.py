from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.preprocessing_service import PreprocessingService

router = APIRouter()

service = PreprocessingService()


class PreprocessingRequest(BaseModel):
    dataset_id: str


@router.post("/preprocess")
def preprocess(request: PreprocessingRequest):

    try:

        result = service.preprocess(
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