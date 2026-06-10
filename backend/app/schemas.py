from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    phone: str = Field(min_length=11, max_length=20)
    password: str = Field(min_length=6, max_length=64)
    confirmPassword: str | None = Field(default=None, min_length=6, max_length=64)
    name: str | None = Field(default=None, max_length=80)


class LoginRequest(BaseModel):
    phone: str = Field(min_length=11, max_length=20)
    password: str = Field(min_length=1, max_length=64)


class WechatLoginRequest(BaseModel):
    code: str = Field(min_length=1)


class UpdateUserRequest(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    avatar: str | None = Field(default=None, max_length=255)
