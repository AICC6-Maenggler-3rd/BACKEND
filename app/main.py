# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from app.api.routers import api_router
from app.db.mongo import connect_to_mongo, close_mongo_connection
from app.db.postgresql import engine, Base

from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.middlewares.log_middleware import UserLogMiddleware
app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.include_router(api_router, prefix="")

@app.get("/")
async def root():
    return {"message": "FastAPI 서버가 정상적으로 실행 중입니다!"}

@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# CORS 설정 (React 개발 서버: http://localhost:5180)
# ⚠️ 순서 중요: CORS는 다른 미들웨어 보다 먼저 또는 나중에 등록해야 함
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 로그 미들웨어 등록
app.add_middleware(UserLogMiddleware)