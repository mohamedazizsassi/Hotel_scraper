import pytest

@pytest.mark.asyncio
async def test_login_success(client):
    resp = await client.post("/auth/login", json={
        "email": "manager@test.com",
        "password": "testpass",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_wrong_password(client):
    resp = await client.post("/auth/login", json={
        "email": "manager@test.com",
        "password": "wrongpass",
    })
    assert resp.status_code == 401

@pytest.mark.asyncio
async def test_login_unknown_email(client):
    resp = await client.post("/auth/login", json={
        "email": "nobody@test.com",
        "password": "testpass",
    })
    assert resp.status_code == 401

@pytest.mark.asyncio
async def test_protected_endpoint_without_token(client):
    resp = await client.get("/manager/calendar")
    assert resp.status_code == 422  # Authorization header is required

@pytest.mark.asyncio
async def test_protected_endpoint_with_invalid_token(client):
    resp = await client.get(
        "/manager/calendar",
        headers={"Authorization": "Bearer not.a.real.token"},
    )
    assert resp.status_code == 401
