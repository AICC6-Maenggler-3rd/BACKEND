# 🧭 SASRec 기반 장소 추천 모델

이 문서는 SASRec(Self-Attentive Sequential Recommendation) 기반 장소 추천 시스템의 구조, 입력/출력, 학습 및 추론 방식을 설명합니다.
본 모델은 사용자의 이전 이동 경로(places), 현재 위치, 동행자 유형(companion) 등의 정보를 활용하여 다음 방문할 장소를 예측합니다.

## 🧩 모델 개요
### 🔹 모델명

SASRec (Self-Attentive Sequential Recommendation)

### 🔹 주요 특징

Transformer 기반 구조: 사용자의 방문 순서를 학습하여 시퀀스 패턴을 모델링.

Self-Attention을 통해 긴 시퀀스 내에서도 중요한 장소 간 관계를 학습.

Sequential Recommendation에 특화되어 있어 ‘다음 장소 예측’에 강점을 가짐.

## 📘 입력 데이터 구성
### 1. places.csv

장소 정보 데이터로, 각 장소의 기본 정보를 포함합니다.

| 컬럼명 | 설명 |
|--------|------|
| `place_id` | 장소 고유 ID |
| `name` | 장소 이름 |
| `lat` | 위도 |
| `lng` | 경도 |
| `category` | 장소 카테고리 (예: 관광지, 식당, 카페 등) |

> `place2idx`, `idx2place` 매핑은 이 CSV 파일로부터 자동 생성됩니다.


### 2. 모델 입력 파라미터

| 파라미터 | 타입 | 설명 |
|-----------|-------|------|
| `model` | str | 학습된 SASRec 모델 파일 경로 (.pt) |
| `places` | list[int] | 이미 확정된 장소 ID 리스트 (예: 여행 일정의 일부) |
| `places_path` | str | `places.csv` 파일 경로 |
| `start_lat`, `start_lng` | float | 현재 혹은 시작 위치 좌표 |
| `companion` | str | 동행자 유형 (예: solo, family, couple 등) |
| `length` | int | 추천받을 장소 개수 |
| `radius_km` | float | 추천 후보 장소의 반경 (km 단위) |
| `step_radius_km` | float | 이전 방문지와 추천지 간 최대 거리 (km 단위) |

## 🧠 모델 구조

```
Input sequence: [p1, p2, p3, ..., pn]
Embedding: item embedding + positional embedding
↓
Self-Attention Layers (num_layers)
↓
Feed-Forward Network
↓
Next-item prediction (Softmax over all items)
```

### 주요 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|-----------|---------|------|
| `hidden_size` | 128 | 임베딩 차원 |
| `num_heads` | 2 | 멀티헤드 어텐션 개수 |
| `num_layers` | 2 | Transformer 인코더 층 수 |
| `max_len` | 20 | 최대 시퀀스 길이 |
---


## ⚙️ 학습(Training)

### 손실 함수
- **Cross-Entropy Loss**
- Positive/Negative sampling을 통한 다음 장소 예측 학습

### 평가 지표
- **Recall@10**  
  사용자가 실제로 다음에 방문한 장소가 모델의 상위 10개 예측 내에 포함되는 비율.

### 학습 루프 예시
```python
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = model.train_step(batch)
    recall10 = evaluate(model, val_loader, top_k=10)
    print(f"Epoch {epoch+1}, Loss={loss:.4f}, Recall@10={recall10:.4f}")
```
## 🚀 추론(Inference)
### 함수: get_next_poi_list()
```
get_next_poi_list(
    model="models/sasrec.pt",
    places=[123, 456],
    places_path="data/places.csv",
    start_lat=37.5665,
    start_lng=126.9780,
    companion="couple",
    length=5,
    radius_km=30.0,
    step_radius_km=10.0
)

리턴값 예시
[872, 913, 885, 1102, 654]
```


모델은 입력된 places 이후 사용자가 방문할 가능성이 높은 장소 ID 리스트를 반환합니다.

## 🧾 처리 흐름

- places.csv 로드 및 place2idx, idx2place 매핑 생성

- 시작 좌표로부터 반경 radius_km 내 후보 장소 필터링

- 모델 입력 시퀀스로 places를 변환

- 학습된 SASRec 모델을 로드하여 다음 장소 확률 분포 예측

- idx2place를 통해 실제 장소 ID로 복원

- 거리 조건(step_radius_km)을 만족하는 상위 장소 반환


### 평가
- Epoch 25 | Loss: 1.6256 | Recall@10: 0.6438