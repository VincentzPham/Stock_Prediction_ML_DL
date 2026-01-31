"""
Root Routes.

Basic API endpoints for health check and welcome message.
"""

from fastapi import APIRouter

from backend.api.schemas import MessageResponse


router = APIRouter(tags=["Root"])


@router.get("/", response_model=MessageResponse)
def read_root() -> MessageResponse:
    """
    API root endpoint.
    
    Returns a welcome message confirming the API is running.
    
    Returns:
        MessageResponse with welcome message.
    """
    return MessageResponse(message="Welcome to Stock Prediction API v2.0")
