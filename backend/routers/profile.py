from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager
from db.models import User
from schemas.profile import ManagerProfile, ProfileUpdate
from services.profile_service import get_profile, update_profile

router = APIRouter(prefix="/manager", tags=["manager"])

@router.get("/me", response_model=ManagerProfile)
async def me_endpoint(
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    return await get_profile(user, db)

@router.patch("/me", response_model=ManagerProfile)
async def update_me_endpoint(
    patch: ProfileUpdate,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    return await update_profile(user, patch, db)
