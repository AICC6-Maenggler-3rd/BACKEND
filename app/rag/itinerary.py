# pip install openai faiss-cpu pandas numpy
import os, json, math
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from openai import OpenAI
from app.core.config import settings
# ---------- 설정 ----------
INDEX_DIR = "faiss_store"                 # index.faiss + meta.json 위치
EMBED_MODEL = "text-embedding-3-small"    # OpenAI 임베딩
EARTH_R = 6371.0088

ALPHA = 0.7          # 텍스트 유사도 가중
TAU_KM = 20.0        # 거리 감쇠 스케일(exp(-d/tau))
CAND_K = 256         # 1차 후보 개수

# ---------- 유틸 ----------
def haversine_km(lat1,lng1,lat2,lng2):
    p=math.pi/180
    a=0.5-math.cos((lat2-lat1)*p)/2+math.cos(lat1*p)*math.cos(lat2*p)*(1-math.cos((lng2-lng1)*p))/2
    return 2*EARTH_R*math.asin(math.sqrt(a))

def geo_weight_km(dist_km, tau=TAU_KM):
    return math.exp(-dist_km/tau)

def load_index_and_meta(index_dir:str):
    # FAISS 인덱스 직접 읽기
    idx = faiss.read_index(f"{index_dir}/index.faiss")
    with open(f"{index_dir}/meta.json","r",encoding="utf-8") as f:
        meta = json.load(f)
    # meta 항목 예시: {"row":0,"id":"123","lat":37.57,"lng":126.98,"name":"경복궁","place_type":"관광지","themes":["문화·예술","관광"]}
    return idx, meta

_client = None
def embed_query(text:str):
    global _client
    if _client is None:
        _client = OpenAI(
  api_key=settings.openai_api_key
)  # OPENAI_API_KEY 필요
    v = _client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    v = np.array(v, dtype="float32").reshape(1,-1)
    faiss.normalize_L2(v)
    return v

def make_query(companion:str, themes:List[str]):
    # 간단 프롬프트 결합
    t = ", ".join(themes) if themes else ""
    return f"여행 동행: {companion}. 선호 테마: {t}. "

def search_candidates(index, meta, query_vec, center_lat, center_lng, radius_km, cand_k=CAND_K):
    D, I = index.search(query_vec, cand_k)
    out=[]
    for j in range(len(I[0])):
        mi = meta[int(I[0][j])]
        if "lat" not in mi or "lng" not in mi: 
            continue
        dist = haversine_km(center_lat, center_lng, mi["lat"], mi["lng"])
        if dist > radius_km:
            continue
        s_text = float(D[0][j])
        s_geo  = geo_weight_km(dist)
        score  = ALPHA*s_text + (1-ALPHA)*s_geo
        out.append({**mi, "dist_km":dist, "s_text":s_text, "s_geo":s_geo, "score":score})
    out.sort(key=lambda x:x["score"], reverse=True)
    return out

def search_candidates_without_position(index, meta, query_vec, cand_k=CAND_K):
    D, I = index.search(query_vec, cand_k)
    out=[]
    for j in range(len(I[0])):
        mi = meta[int(I[0][j])]
        s_text = float(D[0][j])
        score  = ALPHA*s_text
        out.append({**mi,"s_text":s_text, "score":score})
    out.sort(key=lambda x:x["score"], reverse=True)
    return out

def pick_day_route(candidates:List[Dict[str,Any]],
                   center_lat:float, center_lng:float,
                   must_ids:List[str],
                   stops_per_day:int,
                   step_radius_km:float):
    # 1) must 우선
    idset = set()
    chosen=[]
    idstr = lambda v: str(v)
    for p in candidates:
        if idstr(p["id"]) in set(map(idstr, must_ids)):
            chosen.append(p); idset.add(idstr(p["id"]))
    # 2) 나머지 탐욕적 채우기(마지막 지점과의 거리 ≤ step)
    def ok_step(prev, cand):
        if not prev: 
            # 첫 선택은 시작점 근접성 보정
            d0 = haversine_km(center_lat, center_lng, cand["lat"], cand["lng"])
            return True, geo_weight_km(d0, tau=step_radius_km)
        d = haversine_km(prev["lat"], prev["lng"], cand["lat"], cand["lng"])
        return (d <= step_radius_km), geo_weight_km(d, tau=step_radius_km)
    while len(chosen) < stops_per_day - len(must_ids):
        prev = chosen[-1] if chosen else None
        best=None; best_sc=-1.0
        for p in candidates:
            pid=idstr(p["id"])
            if pid in idset: 
                continue
            ok, s_chain = ok_step(prev, p)
            if not ok:
                continue
            sc = 0.85*p["score"] + 0.15*s_chain
            if sc>best_sc:
                best=p; best_sc=sc
        if best is None: 
            break
        chosen.append(best); idset.add(idstr(best["id"]))
    return chosen

# ---------- 장소 검색 함수 ----------
async def search_places(index_dir:str, query:str, count:int, lat:float = None, lng:float = None):
    index, meta = load_index_and_meta(index_dir)
    _query = f"지역명 있으면 무조건 해당 지역만 {query}"
    qv = embed_query(_query)
    
    if lat is not None and lng is not None:
        candidates = search_candidates(index, meta, qv, lat, lng, 50)
    else:
        candidates = search_candidates_without_position(index, meta, qv)
    ids = [candidate["id"] for candidate in candidates]
    return ids[:count]
# ---------- 메인 계획 함수 ----------
async def plan_itinerary(
                   index_dir:str,
                   center_lat:float,
                   center_lng:float,
                   days:int,
                   companion:str,
                   themes:List[str],
                   must_visits_by_day:Optional[Dict[int,List[str]]]=None,  # {1:[id,id], 2:[...]}
                   radius_km:float=30.0,
                   step_radius_km:float=6.0,
                   stops_per_day:int=5) -> List[Dict[str,Any]]:
    """
    반환: [{day:int, items:[{id,name,lat,lng,score,dist_km,...}]}...]
    """
    must_visits_by_day = must_visits_by_day or {}
    index, meta = load_index_and_meta(index_dir)
    qtext = make_query(companion, themes)
    qv = embed_query(qtext)

    # 후보가 부족하면 반경을 단계적으로 확장
    radius_seq = [radius_km, radius_km*1.3, radius_km*1.7, radius_km*2.2]

    # 미리 넉넉히 후보 로드
    for r in radius_seq:
        cand = search_candidates(index, meta, qv, center_lat, center_lng, r, cand_k=CAND_K)
        if len(cand) >= days*stops_per_day:
            candidates = cand; break
    else:
        candidates = cand  # 마지막 반경 결과라도 사용

    # 일자별 구성
    output=[]
    used=set()
    def not_used(p): 
        return str(p["id"]) not in used

    for d in range(1, days+1):
        day_must = must_visits_by_day.get(d, [])  # str 또는 int 혼용 허용
        if len(day_must) >= stops_per_day:
            continue
        # 아직 안 쓴 후보로 필터
        pool = [p for p in candidates if not_used(p) or str(p["id"]) in set(map(str, day_must))]
        
        route = pick_day_route(pool, center_lat, center_lng, day_must, stops_per_day, step_radius_km)
        for p in route: 
            used.add(str(p["id"]))
        # 요약 필드만 출력용으로 축약
        items=[{
            "id":p["id"],
            "name":p.get("name",""),
            "lat":p["lat"], "lng":p["lng"],
            "place_type":p.get("place_type"),
            "themes":p.get("themes"),
            "dist_km":round(p["dist_km"],2),
            "score":round(p["score"],3)
        } for p in route]
        output.append({"day": d, "items": items})
    return output

# ---------- 사용 예 ----------
if __name__ == "__main__":
    plan = plan_itinerary(
        center_lat=37.5324, center_lng=126.9906,    # 지역 좌표
        days=3,                                     # 기간(day)
        companion="가족",                           # 동행
        themes=["문화·예술","관광","먹방"],          # 테마(복수)
        must_visits_by_day={1:["1234"], 3:["987"]}, # 날짜별 필수 여행지
        radius_km=25.0,                             # 검색 반경
        step_radius_km=6.0,                         # 추가 탐색 거리(이동 최대)
        stops_per_day=5
    )
    for day in plan:
        print(f"Day {day['day']}: {[ (it['id'], it['name']) for it in day['items'] ]}")

# ---------- 사용 예 ----------
if __name__ == "__main__":
    plan = plan_itinerary(
        center_lat=37.5324, center_lng=126.9906,    # 지역 좌표
        days=3,                                     # 기간(day)
        companion="가족",                           # 동행
        themes=["문화·예술","관광","먹방"],          # 테마(복수)
        must_visits_by_day={1:["1234"], 3:["987"]}, # 날짜별 필수 여행지
        radius_km=25.0,                             # 검색 반경
        step_radius_km=6.0,                         # 추가 탐색 거리(이동 최대)
        stops_per_day=5
    )
    for day in plan:
        print(f"Day {day['day']}: {[ (it['id'], it['name']) for it in day['items'] ]}")
