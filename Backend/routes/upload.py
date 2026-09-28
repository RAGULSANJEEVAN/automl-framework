from fastapi import APIRouter, UploadFile, File, HTTPException

from services.upload_service import UploadService

router = APIRouter()

service = UploadService()


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...)
):

    try:

        result = service.upload_dataset(file)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )