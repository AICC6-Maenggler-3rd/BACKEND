# 🧭 여행 일정 추천 모델 (Context-GRU4Rec)

## 개요
이 프로젝트는 **여행 일정 추천 시스템**의 핵심 AI 모델로,  
사용자의 여행 흐름(시퀀스)과 맥락(Context)을 함께 고려하여  
“다음에 방문할 장소”를 예측하는 **시퀀스 기반 추천 모델**입니다.

기본 구조는 **GRU4Rec** (Recurrent Neural Network 기반 추천 알고리즘)을 확장한 형태로,  
사용자의 방문 순서 + 여행 정보(동반자, 테마, 거리, 시간 등)를 동시에 학습합니다.

---

## 🧠 GRU4Rec란?

**GRU4Rec**은 2015년 딥러닝 추천 논문 *“Session-based Recommendations with Recurrent Neural Networks”*  
에서 제안된 **세션 기반 추천 알고리즘**입니다.

- 입력: 사용자의 **행동 시퀀스** (예: [A → B → C])  
- 출력: 다음 행동의 확률 분포 (예: D의 확률 ↑)

### 구조

GRU(Gated Recurrent Unit)는 순서를 가진 데이터를 처리하며  
“이전 방문 패턴을 기억하고 다음 행동을 예측”하는 순환 신경망(RNN)의 일종입니다.

### 장점
| 항목 | 설명 |
|------|------|
| 순서 정보 학습 | 사용자의 이동 흐름 반영 |
| 세션 기반 | 로그인 없이도 세션 패턴으로 추천 |
| 학습 속도 빠름 | LSTM보다 가벼움 |
| 단기 + 장기 패턴 | GRU 게이트 구조로 균형 있게 학습 |

---

## 🚀 모델 구조 (Context-GRU4Rec)

기본 GRU4Rec에 **여행 맥락(Context)** 임베딩을 추가했습니다.

```
[장소 ID 시퀀스]
│
├── item_emb (장소 임베딩)
├── comp_emb (동반자)
├── cat_emb (여행 테마)
├── dist_emb (거리 구간)
├── hour_emb (시간대)
▼
[ GRU 층 ]
▼
[ Fully Connected + Softmax ]
▼
다음 장소 확률 예측
```

### 입력 피처
| 피처 | 설명 |
|------|------|
| `item_seq` | 방문한 장소 시퀀스 |
| `companion` | 동반자 (혼자/친구/연인 등) |
| `trip_cats` | 여행 테마 (자연/먹방/관광 등) |
| `distbin` | 시작점 대비 거리 구간 |
| `hour` | 방문 시간대 |

---

## 📊 데이터 구조

### 1. `places_enriched.csv`
| 컬럼 | 설명 |
|------|------|
| place_id | 장소 고유 ID |
| place | 장소명 |
| lat / lng | 위도·경도 |
| place_type | 장소 분류 (관광지, 음식점 등) |
| category | 세부 테마 `[자연,힐링]` |
| popularity | 인기도 점수 |
| region_id | 지역 클러스터 ID |

> 모든 장소의 메타데이터를 포함하는 **아이템 마스터 테이블**

---

### 2. `sessions.csv`
| 컬럼 | 설명 |
|------|------|
| session_id | 여행 세션 ID |
| user_id | 가상 사용자 |
| start_lat / start_lng | 여행 시작 좌표 |
| companion | 동반자 유형 |
| trip_cats | 여행 테마 |
| region_id | 해당 세션의 지역 |

> 하나의 여행(하루 혹은 일정)의 맥락 정보를 담은 **세션 컨텍스트 데이터**

---

### 3. `interactions.csv`
| 컬럼 | 설명 |
|------|------|
| user_id | 사용자 ID |
| session_id | 세션 ID |
| step | 방문 순서 |
| ts | 방문 시각 |
| place_id | 방문 장소 ID |

> 세션 내 실제 방문 순서를 기록한 **시퀀스 로그 데이터**

---

## 🧩 훈련 파이프라인

1. **데이터 로드**
   - `places_enriched`, `sessions`, `interactions` CSV 읽기
2. **시퀀스 생성**
   - `session_id` 기준으로 방문 순서 정렬
3. **입력/라벨 분리**
   - 입력: `[A, B, C]` → 라벨: `D`
4. **모델 학습**
   - CrossEntropyLoss로 “다음 장소 확률” 학습
5. **평가**
   - Recall@10, MRR@10 기준으로 모델 성능 검증

6. **결과**
   - best [EP23] train_loss=2.7507 valid_loss=2.9945 recall@10=0.740 mrr@10=0.545
---

## 🧭 추론 (일정 생성)

입력:
```json
{
  "start_lat": 37.5665,
  "start_lng": 126.9780,
  "companion": "친구와",
  "trip_cats": ["관광", "먹방"],
  "required": [123, 456],
  "length": 7
}
```

과정:

- 시작 좌표 주변 반경 내 후보 장소 필터
- 모델로 각 후보의 “다음 장소 확률” 계산
- 필수 여행지를 중간 슬롯에 삽입
- 반경/거리 제약(radius_km, step_radius_km) 적용
- 최종 일정 반환

출력:

```
[
  {"order": 1, "place": "경복궁", "type": "관광지"},
  {"order": 2, "place": "삼청동 카페거리", "type": "카페"},
  {"order": 3, "place": "북촌 한옥마을", "type": "관광지"},
]
```


## 🧩 주요 하이퍼파라미터
| 이름 | 기본값 | 설명 |
|------|--------|------|
| embedding_dim | 128 | 장소 임베딩 차원 |
| epochs | 25 | 학습 에폭 수 |
| batch_size | 256 | 미니배치 크기 |
| lr | 1e-3 | 학습률 |
| dropout | 0.2 | 드롭아웃 비율 |
| radius_km | 5 | 시작 좌표 반경 |
| step_radius_km | 3 | 연속 장소 간 최대 이동 거리 |
## 🔧 사용 예시
### 학습
```
python train_sequence_recommender.py \
  --places places_enriched.csv \
  --sessions sessions.csv \
  --interactions interactions.csv \
  --out model.pt \
  --epochs 10
```

### 추천(추론)
```
python infer_sequence_recommender.py \
  --model model.pt \
  --places places_enriched.csv \
  --start_lat 37.5665 \
  --start_lng 126.9780 \
  --companion 친구와 \
  --cats "[관광,먹방]" \
  --length 7
```

## 📈 평가 지표
| 지표 | 설명 |
|------|------|
| Recall@K | 상위 K개 예측 중 실제 다음 장소 포함 여부 |
| MRR@K | 실제 정답이 몇 번째로 예측됐는지 평균 역순위 |
| 지리 일관성 | 이동 거리 제약을 벗어나는 비율 |
| 테마 일치율 | 사용자 테마와 추천 장소 카테고리의 교집합 비율 |
## 💡 요약
| 항목 | 설명 |
|------|------|
| 모델명 | Context-GRU4Rec |
| 핵심 | 순서(시퀀스) + 맥락(Context) 학습 |
| 입력 | 방문 장소, 동반자, 테마, 거리, 시간 |
| 출력 | 다음 장소의 확률 |
| 특징 | 반경 기반 필터링 + 필수 여행지 삽입 |
| 응용 | 여행 일정 추천, 사용자 맞춤 경로 생성 |

