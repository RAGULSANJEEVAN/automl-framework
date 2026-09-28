from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.target_service import TargetService

router = APIRouter()

service = TargetService()


class TargetRequest(BaseModel):

    dataset_id: str
    target: str


@router.post("/target")
def select_target(request: TargetRequest):

    try:

        result = service.select_target(
            request.dataset_id,
            request.target
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