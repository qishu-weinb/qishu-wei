import re
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..response import BusinessError, fail, ok
from ..schemas import LoginRequest, RegisterRequest, WechatLoginRequest
from ..security import create_token, hash_password, verify_password
from ..serializers import user_info


router = APIRouter(tags=["auth"])


def validate_phone(phone: str):
    if not re.fullmatch(r"1[3-9]\d{9}", phone):
        raise BusinessError(1001, "手机号格式不正确")


@router.post("/register")
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    validate_phone(payload.phone)
    if payload.confirmPassword is not None and payload.password != payload.confirmPassword:
        raise BusinessError(1001, "两次输入的密码不一致")
    if db.query(User).filter(User.phone == payload.phone).first():
        raise BusinessError(1004, "用户已存在")
    user = User(
        id=str(uuid.uuid4()),
        phone=payload.phone,
        password_hash=hash_password(payload.password),
        name=payload.name or "用户",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return ok(user_info(user), "注册成功")


@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    validate_phone(payload.phone)
    user = db.query(User).filter(User.phone == payload.phone).first()
    if not user:
        raise BusinessError(1002, "用户不存在")
    if not verify_password(payload.password, user.password_hash):
        raise BusinessError(1003, "密码错误")
    return ok({"token": create_token(user.id), "userInfo": user_info(user)}, "登录成功")


@router.post("/logout")
def logout():
    return ok(None, "退出成功")


@router.post("/login/wechat")
def login_wechat(payload: WechatLoginRequest):
    return fail(3002, "微信登录未配置，请使用手机号登录")
