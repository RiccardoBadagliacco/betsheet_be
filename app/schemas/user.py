from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional
from datetime import datetime
import re


class BaseResponse(BaseModel):
    success: bool = Field(...)
    message: str = Field(...)
    data: Optional[dict] = Field(None)
    error: Optional[str] = Field(None)


class ErrorResponse(BaseModel):
    success: bool = Field(False)
    message: str = Field(...)
    error: str = Field(...)
    data: Optional[dict] = Field(None)


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr = Field(...)

    @validator('username')
    def username_valid(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username può contenere solo lettere, numeri e underscore')
        return v.lower()


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(...)

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Le password non coincidono')
        return v


class User(UserBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None


class UserResponse(BaseResponse):
    data: Optional[User] = None
