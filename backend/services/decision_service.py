# backend/services/decision_service.py
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from db.models import User, UserHotelAssignment, ManagerRecommendationDecision
from core.exceptions import NotFoundError
from services.common import get_manager_hotel_id
from schemas.decision import DecisionIn, DecisionBulkIn, DecisionRow

_UPSERT = text("""
    INSERT INTO manager_recommendation_decisions
        (user_id, hotel_id, check_in, nights, adults, boarding_canonical,
         recommended_price_tnd, status)
    VALUES (:uid, :hid, :check_in, :nights, :adults, :boarding, :price, :status)
    ON CONFLICT (user_id, check_in, nights, adults, boarding_canonical)
    DO UPDATE SET status = EXCLUDED.status,
                  recommended_price_tnd = EXCLUDED.recommended_price_tnd,
                  updated_at = now()
""")

async def set_decision(user: User, body: DecisionIn, db: AsyncSession) -> DecisionRow:
    hid = await get_manager_hotel_id(user, db)
    await db.execute(_UPSERT, {
        "uid": str(user.id), "hid": hid, "check_in": body.check_in,
        "nights": body.nights, "adults": body.adults,
        "boarding": body.boarding_canonical, "price": body.recommended_price_tnd,
        "status": body.status,
    })
    await db.commit()
    # Echo the input back as the response: DecisionRow is built from `body`,
    # not re-read from the DB. Acceptable for decision-support tracking — the
    # row we just upserted matches the values we sent (DO UPDATE SET mirrors them).
    return DecisionRow(**body.model_dump())

async def set_decisions_bulk(user: User, body: DecisionBulkIn, db: AsyncSession) -> int:
    hid = await get_manager_hotel_id(user, db)
    for item in body.items:
        await db.execute(_UPSERT, {
            "uid": str(user.id), "hid": hid, "check_in": item.check_in,
            "nights": item.nights, "adults": item.adults,
            "boarding": item.boarding_canonical, "price": item.recommended_price_tnd,
            "status": body.status,
        })
    await db.commit()
    return len(body.items)

async def get_decision_map(user: User, db: AsyncSession) -> dict[tuple, str]:
    result = await db.execute(
        select(
            ManagerRecommendationDecision.check_in,
            ManagerRecommendationDecision.nights,
            ManagerRecommendationDecision.adults,
            ManagerRecommendationDecision.boarding_canonical,
            ManagerRecommendationDecision.status,
        ).where(ManagerRecommendationDecision.user_id == user.id)
    )
    return {(r[0], r[1], r[2], r[3]): r[4] for r in result.fetchall()}
