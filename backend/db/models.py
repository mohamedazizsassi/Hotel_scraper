from __future__ import annotations
import uuid
import datetime
from typing import Optional
from sqlalchemy import String, Boolean, Integer, SmallInteger, DateTime, ForeignKey, Numeric, func, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class City(Base):
    __tablename__ = "cities"
    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=True)
    name_normalized: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    name_display: Mapped[str] = mapped_column(String, nullable=False)


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String)
    role: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True))
    preferences: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb"), default=dict)


class PlatformHotel(Base):
    __tablename__ = "platform_hotels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hotel_name_normalized: Mapped[str] = mapped_column(String, nullable=False)
    hotel_name_display: Mapped[str] = mapped_column(String, nullable=False)
    city_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("cities.id"))
    stars_int: Mapped[Optional[int]] = mapped_column(SmallInteger)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    contact_email: Mapped[Optional[str]] = mapped_column(String)
    contact_phone: Mapped[Optional[str]] = mapped_column(String)


class UserHotelAssignment(Base):
    __tablename__ = "user_hotel_assignments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    hotel_id: Mapped[int] = mapped_column(Integer, ForeignKey("platform_hotels.id"))
    max_competitors: Mapped[int] = mapped_column(SmallInteger, default=4)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class UserCompetitorSelection(Base):
    __tablename__ = "user_competitor_selections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    hotel_id: Mapped[int] = mapped_column(Integer, ForeignKey("platform_hotels.id"))
    display_order: Mapped[int] = mapped_column(SmallInteger)


class ScrapeRun(Base):
    __tablename__ = "scrape_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_ts: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    log_filename: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    source: Mapped[Optional[str]] = mapped_column(String)
    spiders_count: Mapped[int] = mapped_column(Integer, default=0)
    items_total: Mapped[int] = mapped_column(Integer, default=0)
    errors_total: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[Optional[float]] = mapped_column(Numeric)
    status: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PlatformHotelSource(Base):
    __tablename__ = "platform_hotel_sources"
    platform_hotel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("platform_hotels.id"), primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    source_hotel_name: Mapped[str] = mapped_column(String, nullable=False)
    source_city_id: Mapped[Optional[int]] = mapped_column(Integer)
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())


class ManagerRecommendationDecision(Base):
    __tablename__ = "manager_recommendation_decisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    hotel_id: Mapped[int] = mapped_column(Integer, ForeignKey("platform_hotels.id"))
    check_in: Mapped[datetime.date] = mapped_column(nullable=False)
    nights: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    adults: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    boarding_canonical: Mapped[str] = mapped_column(String, nullable=False)
    recommended_price_tnd: Mapped[Optional[float]] = mapped_column(Numeric)
    status: Mapped[str] = mapped_column(String, nullable=False)
    decided_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
