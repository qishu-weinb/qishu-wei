from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class BusinessError(Exception):
    def __init__(self, code: int, message: str, data=None):
        self.code = code
        self.message = message
        self.data = data


def ok(data=None, message="success"):
    return {"code": 0, "message": message, "data": data}


def fail(code: int, message: str, data=None):
    return {"code": code, "message": message, "data": data}


async def business_error_handler(request: Request, exc: BusinessError):
    return JSONResponse(status_code=200, content=fail(exc.code, exc.message, exc.data))


async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=200, content=fail(1001, "参数校验失败", exc.errors()))
