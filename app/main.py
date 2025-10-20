# backend/main.py
from fastapi import FastAPI, UploadFile, File
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
    allow_origins=["http://localhost:5180", "http://192.168.10.220:8381"],
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

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"file_id": file_id, "filename": file.filename, "status": "uploaded"}


@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# 로그 미들웨어 등록
app.add_middleware(UserLogMiddleware)