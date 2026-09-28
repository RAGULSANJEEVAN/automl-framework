from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.analysis_service import AnalysisService
from copy import deepcopy

router = APIRouter()

service = AnalysisService()


class AnalysisRequest(BaseModel):
    dataset_id: str


@router.post("/analyze")
def analyze_dataset(request: AnalysisRequest):

    try:

        analysis = service.analyze_dataset(request.dataset_id)

        response = deepcopy(analysis)

        if "baseline_model" in response:
            if "model_name" in response["baseline_model"]:
                response["baseline_model"]["model_name"] = (
                    response["baseline_model"]["model_name"].__class__.__name__
                )

        return {
            "success": True,
            "data": response
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )