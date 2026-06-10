from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import current_user
from ..models import DiagnosisRecord, User
from ..response import ok
from ..schemas import UpdateUserRequest
from ..serializers import user_info


router = APIRouter(tags=["users"])


@router.get("/user/info")
def get_user_info(user: User = Depends(current_user)):
    return ok(user_info(user))


@router.put("/user/info")
def update_user_info(
    payload: UpdateUserRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    if payload.name is not None:
        user.name = payload.name
    if payload.avatar is not None:
        user.avatar = payload.avatar
    db.commit()
    db.refresh(user)
    return ok(user_info(user), "更新成功")


@router.get("/user/stats")
def get_user_stats(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    base = db.query(DiagnosisRecord).filter(
        DiagnosisRecord.user_id == user.id,
        DiagnosisRecord.deleted_at.is_(None),
    )
    total = base.count()
    benign = base.filter(DiagnosisRecord.result == "benign").count()
    suspicious = base.filter(DiagnosisRecord.result == "malignant").count()
    pending = base.filter(DiagnosisRecord.status == "model_not_configured").count()
    return ok(
        {
            "totalTests": total,
            "benignResults": benign,
            "suspiciousResults": suspicious,
            "pendingResults": pending,
        }
    )
