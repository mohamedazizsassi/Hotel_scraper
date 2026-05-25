# backend/core/exceptions.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class AuthError(Exception):
    def __init__(self, detail: str = "Authentication failed"):
        self.detail = detail

class ForbiddenError(Exception):
    def __init__(self, detail: str = "Access forbidden"):
        self.detail = detail

class NotFoundError(Exception):
    def __init__(self, detail: str = "Not found"):
        self.detail = detail

class MLStoreNotReadyError(Exception):
    def __init__(self, detail: str = "ML models not loaded"):
        self.detail = detail

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuthError)
    async def auth_error_handler(request: Request, exc: AuthError):
        return JSONResponse(status_code=401, content={"detail": exc.detail})

    @app.exception_handler(ForbiddenError)
    async def forbidden_error_handler(request: Request, exc: ForbiddenError):
        return JSONResponse(status_code=403, content={"detail": exc.detail})

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": exc.detail})

    @app.exception_handler(MLStoreNotReadyError)
    async def ml_store_error_handler(request: Request, exc: MLStoreNotReadyError):
        return JSONResponse(status_code=503, content={"detail": exc.detail})
