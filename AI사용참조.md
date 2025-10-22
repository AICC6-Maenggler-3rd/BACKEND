### 🧭 **추천 알고리즘 분류 및 적용 구조**

### 1️⃣ **콘텐츠 기반 필터링 (Content-Based Filtering)**

- **개념**: 장소(POI)의 속성(feature)과 사용자의 선호 속성(feature)을 직접 비교하여 유사도가 높은 항목 추천
- **활용 방식**
  - 장소별 속성: `type`, `theme`, `category`, `평균 체류시간`, `위치좌표(lat, lng)`
  - 사용자 입력: `themes`, `companions`, `travel_style`
  - 예: “힐링·자연 테마”를 선택한 사용자는 `자연`, `산`, `카페` 카테고리 장소를 높은 우선순위로 추천
- **기술 요소**
  - TF-IDF 또는 Word2Vec 기반 카테고리 임베딩
  - Cosine Similarity로 유사도 계산

---

### 2️⃣ **협업 필터링 (Collaborative Filtering)**

- **개념**: 유사한 사용자들의 행동(이용 장소, 일정 선택)을 기반으로 추천
- **활용 방식**
  - 비슷한 일정 패턴(“부모님과”, “2박3일 제주”)을 가진 다른 유저의 일정 데이터를 학습
  - `User-POI` 행렬을 구성하여 ALS(Alternating Least Squares) 또는 Neural CF 모델 적용
- **기술 요소**
  - Latent factor 모델 (Matrix Factorization)
  - Neural Collaborative Filtering (NCF)
  - SASRec, BERT4Rec 같은 시퀀스 기반 CF 모델

---

### 3️⃣ **시퀀스 기반 추천 (Sequential Recommendation)**

- **개념**: 사용자의 장소 방문 순서를 고려하여 다음 장소를 예측
- **활용 방식**
  - `next_poi_model`로 이전 방문 장소 시퀀스를 입력받아 다음 POI 예측
  - Transformer 기반 SASRec 모델 활용 (Self-Attention으로 순서 의존성 학습)
- **기술 요소**
  - SASRec (Self-Attentive Sequential Recommendation)
  - GRU4Rec (RNN 기반)
  - BERT4Rec (Bidirectional Transformer 기반)

---

### 4️⃣ **하이브리드 필터링 (Hybrid Filtering)**

- **개념**: 콘텐츠 기반 + 협업 필터링 + 시퀀스 기반을 통합하여 정확도 향상
- **활용 방식**
  - 1차 필터: 콘텐츠 기반으로 후보군 축소
  - 2차 랭킹: 협업 필터링 점수와 시퀀스 확률을 가중 평균
  - 3차 후처리: 시간·거리 제약 조건 필터링
- **기술 요소**
  - Weighted Hybrid Model
  - Meta-learner (LightGBM, XGBoost 등으로 세 점수 통합)

---

### 5️⃣ **지리·시간 제약 기반 필터링 (Spatiotemporal Filtering)**

- **개념**: 장소 간 거리, 이동 시간, 영업시간 등을 제약 조건으로 고려
- **활용 방식**
  - 지도 좌표 기반 거리 계산(`haversine_distance`)
  - 일정 생성 시 `start_time`, `end_time` 자동 조정
- **기술 요소**
  - Geo-aware Recommendation
  - Route Optimization (Dijkstra, A\*)
  - Time-window filtering (open/close_hour 기반)

---

### 6️⃣ **강화학습 기반 추천 (Reinforcement Learning-based)**

- **개념**: 일정 추천을 ‘경로 최적화 문제’로 보고, 보상(reward)을 학습
- **활용 방식**
  - 보상 = “사용자 만족도 + 이동 효율성 + 테마 일치도”
  - 에피소드 단위로 일차별 일정 완성
- **기술 요소**
  - Deep Q-Network (DQN)
  - Policy Gradient / PPO

---

### ⚙️ **시스템에 추천되는 통합 구조**

| 단계           | 알고리즘 유형             | 역할                             |
| -------------- | ------------------------- | -------------------------------- |
| 후보 장소 선정 | 콘텐츠 기반 필터링        | 테마·카테고리 기반 1차 필터링    |
| 장소 순서 추천 | 시퀀스 기반 추천 (SASRec) | 장소 방문 순서 예측              |
| 개인화 보정    | 협업 필터링 (NCF)         | 유사 사용자 일정 패턴 반영       |
| 일정 후처리    | 지리·시간 제약 필터링     | 거리·이동시간·운영시간 제약 보정 |
| 지속 개선      | 강화학습 기반 추천        | 사용자 피드백 기반 보상 학습     |
