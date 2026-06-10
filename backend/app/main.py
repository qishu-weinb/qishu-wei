from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import API_PREFIX, UPLOAD_DIR
from .database import Base, SessionLocal, engine
from .response import BusinessError, business_error_handler, validation_error_handler
from .routers import auth, history, knowledge, upload, users
from .seed import seed_knowledge


def create_app() -> FastAPI:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_knowledge(db)
    finally:
        db.close()

    app = FastAPI(title="乳腺癌病理组织判断后端")
    app.add_exception_handler(BusinessError, business_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(users.router, prefix=API_PREFIX)
    app.include_router(upload.router, prefix=API_PREFIX)
    app.include_router(history.router, prefix=API_PREFIX)
    app.include_router(knowledge.router, prefix=API_PREFIX)
    return app


app = create_app()
