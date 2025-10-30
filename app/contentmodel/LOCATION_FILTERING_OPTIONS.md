# 지역 필터링 방식 비교

## 현재 구현: 좌표 기반 필터링 ✅ (권장)
장점:
- ✅ 정확한 지리적 거리 계산 (Haversine 공식)
- ✅ Region 테이블 활용으로 관리 용이
- ✅ 반경 조절 가능 (기본 50km)
- ✅ 키워드 폴백 지원

단점:
- ⚠️ Region 데이터가 많아야 함

---

## 대안 1: 좌표 범위(경계) 기반 필터링
지역의 사각형 경계 좌표로 필터링

```python
async def _filter_places_by_bounding_box(
    self, 
    places_data: List[Dict], 
    location: str, 
    db: AsyncSession,
    north: float, south: float, east: float, west: float
) -> List[Dict]:
    """경계 좌표로 장소 필터링"""
    filtered_places = []
    for place in places_data:
        place_lat = place.get('address_la')
        place_lng = place.get('address_lo')
        
        if (south <= place_lat <= north) and (west <= place_lng <= east):
            filtered_places.append(place)
    
    return filtered_places
```

장점: DB 쿼리 최적화 가능
단점: 경계 좌표 데이터 필요

---

## 대안 2: 외부 API 활용 (Kakao/Naver Geocoding)

```python
import httpx

async def get_location_coordinates(location_name: str, api_key: str) -> tuple:
    """Kakao API로 지역 좌표 조회"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": location_name}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        data = response.json()
        
        if data.get('documents'):
            first_result = data['documents'][0]
            lng = float(first_result['x'])
            lat = float(first_result['y'])
            return lat, lng
    
    return None, None
```

장점: 자동 업데이트, 대량 데이터
단점: API 호출 비용/지연, 제한 존재

---

## 대안 3: 혼합 방식 (하이브리드)
좌표 기반 + 키워드 + 거리 가중치

```python
async def _filter_places_hybrid(
    self, 
    places_data: List[Dict], 
    location: str, 
    db: AsyncSession,
    radius_km: float = 50.0,
    keyword_weight: float = 0.3
) -> List[Dict]:
    """혼합 방식: 좌표 + 키워드 + 가중치"""
    
    # 1. 좌표 기반 필터링
    coord_filtered = await self._filter_places_by_location(
        places_data, location, db, radius_km
    )
    
    # 2. 키워드 기반 필터링
    keyword_filtered = self._filter_places_by_keyword(places_data, location)
    
    # 3. 가중치 결합
    combined_places = {}
    
    for place in coord_filtered:
        place_id = place['place_id']
        combined_places[place_id] = {
            'place': place,
            'score': (1 - keyword_weight)  # 거리 기반 점수
        }
    
    for place in keyword_filtered:
        place_id = place['place_id']
        if place_id in combined_places:
            combined_places[place_id]['score'] += keyword_weight
        else:
            combined_places[place_id] = {
                'place': place,
                'score': keyword_weight  # 키워드 점수만
            }
    
    # 점수 순 정렬
    result = sorted(combined_places.values(), key=lambda x: x['score'], reverse=True)
    return [item['place'] for item in result]
```

장점: 정확도 높음, 유연함
단점: 복잡도 높음

---

## 추천 사항

### 🏆 현재 구현 (좌표 기반) 유지
가장 균형잡힌 방식입니다.

### 추가 개선 아이디어:
1. **캐싱**: Region 데이터를 인메모리 캐시하여 DB 조회 최소화
2. **적응형 반경**: 지역 크기에 따라 반경 자동 조절
   - 대도시(서울, 부산): 30km
   - 중소도시: 20km  
   - 시골/섬(제주): 50km
3. **사용자 피드백**: 필터링 결과가 너무 적으면 반경 자동 확장

