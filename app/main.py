# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from app.api.routers import api_router
from app.db.mongo import connect_to_mongo, close_mongo_connection
from app.db.postgresql import engine, Base

from app.middlewares.log_middleware import UserLogMiddleware
app = FastAPI()



# CORS 설정 (React 개발 서버: http://localhost:5180)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5180", "http://192.168.10.220:8381", "https://inpick.aicc-project.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

# 로그 미들웨어 등록
app.add_middleware(UserLogMiddleware)