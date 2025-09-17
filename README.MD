## 파일 구조

  project_name/
  │── alembic/                # DB 마이그레이션 (Alembic 사용 시)
  │── app/
  │   ├── api/                # API 엔드포인트 (Router)
  │   │   ├── endpoints/  # 세부 엔드포인트 파일
  │   │   │   ├── users.py
  │   │   │   ├── auth.py
  │   │   │   └── items.py
  │   │   └── router.py   #전체 router 묶음
  │   │
  │   ├── core/               # 핵심 설정
  │   │   ├── config.py       # 설정 (환경변수, pydantic BaseSettings)
  │   │   ├── security.py     # 보안 관련 (JWT, Password hashing)
  │   │   └── logging.py      # 로깅 설정
  │   │
  │   ├── db/
  │   │   ├── base.py         # Base 클래스 (SQLAlchemy)
  │   │   └── session.py      # DB 세션 연결
  │   │
  │   ├── models/             # SQLAlchemy 모델
  │   │
  │   ├── schemas/            # Pydantic 스키마
  │   │
  │   ├── services/           # 비즈니스 로직 (서비스 계층)
  │   │
  │   ├── repositories/       # DB 접근 계층 (CRUD 함수)
  │   │
  │   ├── utils/              # 유틸 함수 (공통 모듈)
  │   │
  │   └── main.py             # FastAPI 엔트리포인트
  │
  ├── .env                    # 환경 변수
  └── requirements.txt        # 의존성 패키지

## 로컬 실행
npm run dev