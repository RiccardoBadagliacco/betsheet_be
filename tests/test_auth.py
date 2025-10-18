import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import Base, engine, SessionLocal
from app.db.models import User
from app.core.security import get_password_hash


@pytest.fixture(scope="function")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


def test_login_and_me(client):
    # prepare a user
    db = SessionLocal()
    pw = get_password_hash("StrongPass1")
    user = User(username="testuser", email="test@example.com", hashed_password=pw)
    db.add(user)
    db.commit()
    db.refresh(user)

    resp = client.post("/auth/login", json={"username": "testuser", "password": "StrongPass1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    token = data["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200
