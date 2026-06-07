import uuid
import pytest
from db.models import User
from core.dependencies import get_current_admin
from core.exceptions import ForbiddenError


def _user(role: str) -> User:
    return User(id=uuid.uuid4(), email="x@y.tn", password_hash="h", full_name="X",
                role=role, is_active=True)


async def test_get_current_admin_allows_admin():
    u = _user("admin")
    assert await get_current_admin(u) is u


async def test_get_current_admin_rejects_manager():
    with pytest.raises(ForbiddenError):
        await get_current_admin(_user("manager"))
