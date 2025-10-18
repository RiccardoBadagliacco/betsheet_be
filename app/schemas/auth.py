from pydantic import BaseModel
from typing import Optional


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    username: Optional[str] = None


class LoginPayload(BaseModel):
    username: str
    password: str


class RefreshPayload(BaseModel):
    refresh_token: str
