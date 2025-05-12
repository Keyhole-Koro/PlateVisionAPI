from fastapi import FastAPI
from app.routes.process_image import router as process_image_router
from app.routes.ping import router as ping_router

app = FastAPI(debug=True)

@app.on_event("startup")
async def startup_event():
    pass
    # Load models

app.include_router(process_image_router)
app.include_router(ping_router)