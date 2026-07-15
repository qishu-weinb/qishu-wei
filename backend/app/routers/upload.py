import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from ..config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_SIZE, UPLOAD_DIR
from ..database import get_db
from ..dependencies import current_user
from ..models import DiagnosisRecord, PathologyImage, User
from ..response import fail


router = APIRouter(tags=["upload"])


@router.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return fail(2002, "图片格式不支持，仅支持 jpg、png")
    content = await file.read()
    if not content:
        return fail(2001, "文件上传失败")
    if len(content) > MAX_UPLOAD_SIZE:
        return fail(2001, "图片大小不能超过5MB")

    image_id = str(uuid.uuid4())
    suffix = ALLOWED_IMAGE_TYPES[file.content_type]
    filename = f"{image_id}{suffix}"
    target = UPLOAD_DIR / filename
    target.write_bytes(content)

    image = PathologyImage(
        id=image_id,
        user_id=user.id,
        original_name=Path(file.filename or filename).name,
        file_path=str(target),
        file_url=f"/uploads/{filename}",
        mime_type=file.content_type,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    record = DiagnosisRecord(
        id=str(uuid.uuid4()),
        user_id=user.id,
        image_id=image.id,
        status="model_not_configured",
        analysis="AI模型未配置，暂不能生成诊断结果。",
        suggestion="请配置训练完成的模型或外部推理服务后重新进行诊断。",
    )
    db.add(image)
    db.add(record)
    db.commit()

    return fail(
        3001,
        "AI模型未配置，暂不能生成诊断结果",
        {
            "recordId": record.id,
            "imageId": image.id,
            "imageUrl": image.file_url,
            "status": record.status,
            "analysis": record.analysis,
            "suggestion": record.suggestion,
        },
    )
