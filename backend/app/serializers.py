from datetime import datetime

from .models import DiagnosisRecord, KnowledgeArticle, User


def fmt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def user_info(user: User):
    return {
        "id": user.id,
        "phone": user.phone or "",
        "name": user.name,
        "avatar": user.avatar,
        "createdAt": fmt(user.created_at),
        "updatedAt": fmt(user.updated_at),
    }


def record_item(record: DiagnosisRecord):
    result_label = "待诊断"
    if record.result == "benign":
        result_label = "良性检测结果"
    elif record.result == "malignant":
        result_label = "恶性检测结果"
    return {
        "id": record.id,
        "imageId": record.image_id,
        "imageUrl": record.image.file_url if record.image else "",
        "status": record.status,
        "result": record.result,
        "resultLabel": result_label,
        "hasCancer": record.result == "malignant" if record.result else False,
        "confidence": round(record.confidence * 100) if record.confidence is not None else 0,
        "confidenceRaw": record.confidence,
        "analysis": record.analysis,
        "suggestion": record.suggestion,
        "createdAt": fmt(record.created_at),
    }


def article_list_item(article: KnowledgeArticle):
    return {
        "id": article.id,
        "tag": article.tag,
        "title": article.title,
        "summary": article.summary,
        "cover": article.cover,
        "createdAt": article.published_at.strftime("%Y-%m-%d"),
    }


def article_detail(article: KnowledgeArticle):
    data = article_list_item(article)
    data["content"] = article.content
    data["date"] = article.published_at.strftime("%Y-%m-%d")
    return data
