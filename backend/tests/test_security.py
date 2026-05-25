import uuid
import datetime
import pytest
import jwt
from core.security import hash_password, verify_password, create_access_token, decode_access_token

def test_password_hash_and_verify():
    h = hash_password("my_secret")
    assert verify_password("my_secret", h)
    assert not verify_password("wrong", h)

def test_password_hashes_are_unique():
    h1 = hash_password("same")
    h2 = hash_password("same")
    assert h1 != h2  # bcrypt uses a unique salt each time

def test_jwt_round_trip():
    uid = str(uuid.uuid4())
    token = create_access_token(uid, hotel_id=7, role="manager")
    payload = decode_access_token(token)
    assert payload["sub"] == uid
    assert payload["hotel_id"] == 7
    assert payload["role"] == "manager"

def test_jwt_admin_has_no_hotel():
    uid = str(uuid.uuid4())
    token = create_access_token(uid, hotel_id=None, role="admin")
    payload = decode_access_token(token)
    assert payload["hotel_id"] is None
    assert payload["role"] == "admin"

def test_jwt_expired_raises(monkeypatch):
    past = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    monkeypatch.setattr("core.security._now", lambda: past)
    token = create_access_token("u", hotel_id=1, role="manager")
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token)
