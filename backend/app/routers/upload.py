import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from ..ai_inference import infer_image, model_status
from ..config import ALLOWED_IMAGE_TYPES, MAX_UPLOAD_SIZE, MODEL_VERSION, UPLOAD_DIR
from ..database import get_db
from ..dependencies import current_user
from ..models import DiagnosisRecord, PathologyImage, User
from ..response import BusinessError, ok


router = APIRouter(tags=["upload"])
logger = logging.getLogger(__name__)


@router.get("/model/health")
def model_health():
    return ok(model_status())


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

    try:
        prediction = infer_image(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        logger.exception("model inference failed")
        raise BusinessError(5001, "模型推理失败，请检查后端模型依赖和权重文件") from exc

    mask_filename = f"{image_id}-mask.png"
    mask_target = UPLOAD_DIR / mask_filename
    Image.fromarray(prediction["mask"]).save(mask_target)
    result = prediction["result"]
    confidence = prediction["confidence"]
    if result == "malignant":
        analysis = "模型在当前图像中识别到更明显的恶性倾向。"
        suggestion = "请尽快携带原始影像和检查资料咨询乳腺专科医生，必要时进行病理检查。"
    elif result == "benign":
        analysis = "模型更倾向于良性病灶。"
        suggestion = "模型结果不能排除疾病，请结合超声报告、BI-RADS分级和医生意见判断。"
    else:
        analysis = "模型未在当前图像中识别到明确病灶。"
        suggestion = "如存在症状或医生建议，请继续完成正规影像学检查和随访。"
    disclaimer = "本结果仅供科研和辅助参考，不能替代医生诊断。"
    analysis = f"{analysis}{disclaimer}"

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
        status="completed",
        result=result,
        confidence=confidence,
        analysis=analysis,
        suggestion=suggestion,
        model_version=MODEL_VERSION,
    )
    db.add(image)
    db.add(record)
    db.commit()

    return ok(
        {
            "recordId": record.id,
            "imageId": image.id,
            "imageUrl": image.file_url,
            "maskUrl": f"/uploads/{mask_filename}",
            "status": record.status,
            "result": result,
            "resultLabel": {"normal": "未见明显病灶", "benign": "良性倾向", "malignant": "恶性倾向"}[result],
            "confidence": round(confidence * 100),
            "confidenceRaw": confidence,
            "probabilities": {key: round(value * 100, 2) for key, value in prediction["probabilities"].items()},
            "analysis": record.analysis,
            "suggestion": record.suggestion,
            "modelVersion": MODEL_VERSION,
            "device": prediction["device"],
        },
        "检测完成",
    )
