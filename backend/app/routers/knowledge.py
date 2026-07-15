from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import KnowledgeArticle
from ..response import BusinessError, ok
from ..serializers import article_detail, article_list_item


router = APIRouter(tags=["knowledge"])


@router.get("/knowledge")
def get_knowledge_list(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=50),
    keyword: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(KnowledgeArticle)
    if keyword:
        like = f"%{keyword}%"
        query = query.filter(
            (KnowledgeArticle.title.like(like))
            | (KnowledgeArticle.summary.like(like))
            | (KnowledgeArticle.content.like(like))
        )
    query = query.order_by(KnowledgeArticle.published_at.desc())
    total = query.count()
    articles = query.offset((page - 1) * size).limit(size).all()
    return ok({"list": [article_list_item(item) for item in articles], "total": total, "page": page, "size": size})


@router.get("/knowledge/{article_id}")
def get_knowledge_detail(article_id: str, db: Session = Depends(get_db)):
    article = db.get(KnowledgeArticle, article_id)
    if not article:
        raise BusinessError(404, "文章不存在")
    return ok(article_detail(article))
