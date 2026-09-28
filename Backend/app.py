from fastapi import FastAPI

from routes.upload import router as upload_router
from routes.target_routes import router as target_router
from routes.analysis import router as analysis_router
from routes.recommendation import router as recommendation_router
from routes.preprocessing import router as preprocessing_router
from routes.training import router as training_router
# from ai.report_api import router as report_router
from routes.dataset import router as dataset_router
from ai_report.api.report_router import router as report_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="AutoML API",
    version="1.0"
)

app.include_router(
    upload_router,
    prefix="/api"
)

app.include_router(
    target_router,
    prefix="/api"
)

app.include_router(
    analysis_router,
    prefix="/api"
)

app.include_router(
    recommendation_router,
    prefix="/api"
)

app.include_router(
    preprocessing_router,
    prefix="/api"
)


app.include_router(
    training_router,
    prefix="/api",
    tags=["Training"]
)

# app.include_router(
#     report_router,
#     prefix="/api"
# )

app.include_router(
    dataset_router,
    prefix="/api",
    tags=["Dataset"]
)

app.include_router(report_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:64686",
        "http://localhost:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "AutoML API Running"
    }