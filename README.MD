# 🗺️ Inpick BACKEND

AI 기반 **여행 일정 생성 및 추천 시스템**의 백엔드 서비스입니다.  
사용자의 여행지, 동행자, 테마, 위치 정보를 기반으로 개인화된 일정을 생성하고  
콘텐츠 기반 추천, 협업 필터링, 시퀀스 기반 Next-POI, RAG(Vector DB)를 활용한 장소 추천을 제공합니다.

---

## 🚀 주요 기능

- 🔐 **소셜 로그인** – Google / Naver / Kakao OAuth 인증
- 🧭 **여행 일정 관리** – 일정 생성, 수정, 조회, 삭제
- 🏝️ **추천 시스템**
  - 콘텐츠 기반 추천 (CBF)
  - 협업 필터링 (CF)
  - 시퀀스 기반 추천 (GRU4Rec, SASRec)
  - RAG 기반 장소 검색 (FAISS)
- 🧑‍💼 **관리자 페이지** – 사용자 관리, 로그 관리, 카테고리 관리
- 📊 **활동 로그** – 모든 주요 이벤트 로깅

---

## 🧩 기술 스택

| 분류                 | 기술                             |
| -------------------- | -------------------------------- |
| **Language**         | Python                           |
| **Framework**        | FastAPI                          |
| **Database**         | PostgreSQL, MongoDB              |
| **Vector DB**        | FAISS                            |
| **Machine Learning** | PyTorch, Implicit (ALS)          |
| **LLM/AI**           | OpenAI, Claude (via LangChain)   |
| **Auth**             | OAuth 2.0 (Google, Naver, Kakao) |
| **Map API**          | NCP Map, TMAP API                |
| **Infra**            | Docker, Jenkins, Gunicorn        |
| **Env**              | Conda (`environment.yml`)        |

---

## 📁 디렉토리 구조
```
app/
├─ api/
│ ├─ endpoint/ # 주요 API 엔드포인트
│ │ ├─ auth.py # 로그인 / 콜백
│ │ ├─ itinerary.py # 일정 관리
│ │ ├─ recommendation.py # 추천 기능
│ │ ├─ place.py, region.py, map.py
│ │ └─ manage.py # 관리자 기능
│ └─ routers.py # 라우터 묶음
├─ auth/
│ ├─ base.py # OAuth 추상 클래스
│ ├─ dependencies.py # 세션 인증 의존성
│ ├─ google.py, naver.py, kakao.py
├─ services/
│ ├─ activity_log_service.py # 활동 로그
│ ├─ generate_itinerary_service.py # 일정 생성 (모델별)
│ ├─ itinerary_service.py # 일정 CRUD
│ ├─ map_path_service.py # 경로 탐색
│ ├─ session_service.py # 세션 관리
│ ├─ user_service.py # 사용자 관리
│ └─ log_service.py # 로깅
├─ repositories/
│ ├─ accommodationdb.py # 숙소 DB 접근
│ ├─ categorydb.py # 카테고리 DB 접근
│ ├─ placedb.py # 장소 DB 접근
│ ├─ regiondb.py # 지역 DB 접근
│ └─ userdb.py # 사용자 DB 접근
├─ contentmodel/ # 콘텐츠 기반 추천 모듈
├─ core/ # 환경설정 및 로깅
├─ db/ # DB 연결 (Postgres, Mongo)
├─ middlewares/ # 로깅 미들웨어
├─ ml/ # ML 모델 (ALS, GRU4Rec, SASRec 등)
├─ models/ # SQLAlchemy ORM
├─ rag/ # RAG + FAISS 벡터 검색
├─ schemas/ # Pydantic 스키마 정의
└─ main.py # FastAPI 진입점

deploy/
├─ dockerfile
└─ jenkinsfile

environment.yml
package.json
README.md
```

---

## 🧠 시스템 구성

1️⃣ 인증 플로우  
사용자 → /auth/{provider}/login  
→ OAuth Provider 인증  
→ /auth/{provider}/callback  
→ 사용자 정보 검증  
→ **MongoDB 세션 생성 (TTL: 1800초)**  
→ app_user 테이블 등록/갱신  
→ **쿠키 기반 세션 관리**

**세션 재활성화**: 매 요청마다 자동 갱신 (UserLogMiddleware)
**로그아웃**: GET /auth/logout (세션 삭제)
**계정 삭제**: DELETE /auth/user (soft delete)

2️⃣ 추천 시스템 구성  
| 방식 | 경로 | 설명 |
|------|------|------|
| CBF | app/contentmodel/content_based_recommendation_service.py | 카테고리, 테마, 설명 기반 코사인 유사도 |
| CF (ALS) | app/ml/collaborativeFiltering/main.py | 사용자–아이템 행렬 분해 |
| Next-POI (GRU4Rec / SASRec) | app/ml/next_poi_gru4rec.py, app/ml/next_poi_sas_rec.py | 시퀀스 기반 다음 장소 예측 |
| RAG | app/rag/faiss_store/ | 위치·테마·동행자 조건으로 벡터 검색 |

---

## ⚙️ 실행 방법

1️⃣ 환경 구성  
conda env create -f environment.yml  
conda activate inpick-backend

2️⃣ 환경 변수 설정 (.env 예시)  
DATABASE_URL=postgresql+psycopg2://USER:PASS@HOST:PORT/DBNAME  
MONGO_URI=mongodb://USER:PASS@HOST:PORT/DBNAME

GOOGLE_CLIENT_ID=...  
GOOGLE_CLIENT_SECRET=...  
NAVER_CLIENT_ID=...  
NAVER_CLIENT_SECRET=...  
KAKAO_CLIENT_ID=...  
KAKAO_CLIENT_SECRET=...

OAUTH_REDIRECT_BASE=https://your.domain/auth/callback  
APP_ENV=local  
LOG_LEVEL=INFO

실제 키명은 app/core/config.py 참고

3️⃣ 서버 실행  
uvicorn app.main:app --reload --port 8000

Docs: http://localhost:8000/docs

---

## 🧪 API 예시

📍 장소 CBF 추천 예시  
curl -X GET "http://localhost:8000/recommendation/places?address=제주&suitables[]=자연&categorys[]=관광&center_location[]=33.4&center_location[]=126.5"

📍 장소 ALS 추천 예시  
curl -X GET "http://localhost:8000/recommendation/places?address=제주&suitables[]=자연&categorys[]=관광&center_location[]=33.4&center_location[]=126.5"

🧭 일정 생성 예시 (Next-POI, SASRec, RAG 포함)
curl -X POST http://localhost:8000/itinerary/generate \
 -H "Content-Type: application/json" \
 -d '{
"model_name": "맞춤형 일정 추천", // 또는 "실시간 관심 기반 추천" / "GPT 추천"
"base_itinerary": {...}
}'

📋 사용 가능한 모델 목록
curl -X GET http://localhost:8000/itinerary/models

---

## 🐳 Docker 실행

docker build -t inpick-backend:latest -f deploy/dockerfile .  
docker run -p 8000:8000 --env-file .env inpick-backend:latest

---

## 🔄 CI/CD (Jenkins)

deploy/jenkinsfile 기반 파이프라인  
Build → Test → Dockerize → Deploy

포트 충돌, 헬스체크, 환경변수 자동 로드 포함

---

## 🧱 데이터 모델 요약

| 테이블              | 설명                                     |
| ------------------- | ---------------------------------------- |
| app_user            | 유저 계정, 소셜 로그인, soft delete 지원 |
| place               | 장소(관광지, 음식점, 카페 등)            |
| accommodation       | 숙소 정보                                |
| itinerary           | 여행 일정(시작/종료일, 구성원, 테마 등)  |
| itinerary_item      | 일정 내 세부 장소                        |
| category            | 장소 카테고리                            |
| **region**          | **지역 정보 (위경도 포함)**              |
| **place_category**  | **장소-카테고리 연결 (다대다)**          |
| **place_registrar** | **사용자 등록 장소 (다대다)**            |

---

## 📋 체크리스트

✅ .env 환경 변수 세팅 완료  
✅ DB 연결 확인  
✅ FAISS 인덱스 파일 존재 (app/rag/faiss_store/)  
✅ PyTorch 모델 로드 가능 (app/ml/artifacts/\*.pt)  
✅ OAuth Redirect URI 일치  
✅ 로그 기록 정상 작동

---

## 🪪 라이선스

본 프로젝트의 저작권은 **AICC6-Maenggler-3rd 팀**에 있으며,  
상용 및 배포 정책은 별도 라이선스 조항을 따릅니다.

---

## 👥 팀 정보

**AICC6-Maenggler-3rd Team**  
AI 기반 여행 일정 생성 서비스 “Inpick”
