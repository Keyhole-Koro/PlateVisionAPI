from fastapi import FastAPI
from services.model_loader import load_initial_models
from app.routes.process_image import router as process_image_router

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.models = await load_initial_models()

app.include_router(process_image_router)
