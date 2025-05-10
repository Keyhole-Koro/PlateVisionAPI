from fastapi import APIRouter

router = APIRouter()

@router.post("/ping/")
def ping():
    """
    Ping the server to check if it's running.
    """
    return {"message": "Pong!"}