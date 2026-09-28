from fastapi import APIRouter, HTTPException

from services.dataset_service import DatasetService

router = APIRouter()

service = DatasetService()


@router.get("/dataset/{dataset_id}")
def get_dataset(dataset_id: str):

    try:

        result = service.get_dataset(dataset_id)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )