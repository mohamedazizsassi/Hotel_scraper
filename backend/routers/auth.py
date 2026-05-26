from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db
from schemas.auth import LoginRequest, TokenResponse
from services.auth_service import login

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=TokenResponse)
async def login_endpoint(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    token = await login(body.email, body.password, db)
    return TokenResponse(access_token=token)
