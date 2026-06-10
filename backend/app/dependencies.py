from fastapi import Depends, Header
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .response import BusinessError
from .security import decode_token


def current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise BusinessError(401, "请先登录")
    token = authorization.removeprefix("Bearer ").strip()
    payload = decode_token(token)
    user = db.get(User, payload.get("sub"))
    if not user:
        raise BusinessError(401, "登录已过期")
    return user
