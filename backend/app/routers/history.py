from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload

from ..database import get_db
from ..dependencies import current_user
from ..models import DiagnosisRecord, User
from ..response import BusinessError, ok
from ..serializers import record_item


router = APIRouter(tags=["history"])


def user_record_query(db: Session, user: User):
    return (
        db.query(DiagnosisRecord)
        .options(joinedload(DiagnosisRecord.image))
        .filter(DiagnosisRecord.user_id == user.id, DiagnosisRecord.deleted_at.is_(None))
    )


@router.get("/history")
def get_history(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=50),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    query = user_record_query(db, user).order_by(DiagnosisRecord.created_at.desc())
    total = query.count()
    list_data = query.offset((page - 1) * size).limit(size).all()
    return ok({"list": [record_item(item) for item in list_data], "total": total, "page": page, "size": size})


@router.get("/history/{record_id}")
def get_history_detail(
    record_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    record = user_record_query(db, user).filter(DiagnosisRecord.id == record_id).first()
    if not record:
        raise BusinessError(404, "记录不存在")
    return ok(record_item(record))


@router.get("/diagnosis/{image_id}")
def get_diagnosis_by_image(
    image_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    record = user_record_query(db, user).filter(DiagnosisRecord.image_id == image_id).first()
    if not record:
        raise BusinessError(404, "诊断结果不存在")
    return ok(record_item(record))


@router.delete("/history/{record_id}")
def delete_history(
    record_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    record = user_record_query(db, user).filter(DiagnosisRecord.id == record_id).first()
    if not record:
        raise BusinessError(404, "记录不存在")
    record.deleted_at = datetime.now()
    db.commit()
    return ok(None, "删除成功")
