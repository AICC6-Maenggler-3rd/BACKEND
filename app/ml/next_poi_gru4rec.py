# infer_sequence_recommender.py
# pip install torch pandas numpy

import argparse, math
import numpy as np, pandas as pd
import torch, torch.nn as nn
from math import radians, sin, cos, asin, sqrt
# ==== 고정 마스터 값 ====
CATS = ["액티비티","자연","바다","산","쇼핑","문화예술","관광","먹방","힐링"]
COMPANIONS = ["혼자서","친구와","연인과","배우자와","부모님과","아이와"]
COMP_IDX = {c:i for i,c in enumerate(COMPANIONS)}
CAT_IDX  = {c:i for i,c in enumerate(CATS)}

# ==== 유틸 ====
def parse_cats(cat_str):
    s = str(cat_str).strip()
    if s.startswith("[") and s.endswith("]"): s = s[1:-1]
    return [c for c in s.split(",") if c]

def haversine(lat1,lng1,lat2,lng2):
    R=6371.0
    p1=np.deg2rad(lat1); p2=np.deg2rad(lat2)
    dlat=p2-p1; dlng=np.deg2rad(lng2-lng1)
    a=np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlng/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def bin_distance_km(d):
    if d<=1: return 0
    if d<=2: return 1
    if d<=3: return 2
    if d<=5: return 3
    if d<=8: return 4
    if d<=13: return 5
    return 6

# ==== 모델 (학습 코드와 동일) ====
class ContextGRURec(nn.Module):
    def __init__(self, n_items, d=128, n_comp=6, n_cat=9, n_distbin=7, n_hour=24, pdrop=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+1, d, padding_idx=0)
        self.comp_emb = nn.Embedding(n_comp, 16)
        self.cat_emb  = nn.Embedding(n_cat, 16)
        self.dist_emb = nn.Embedding(n_distbin, 8)
        self.hour_emb = nn.Embedding(n_hour, 8)
        self.gru = nn.GRU(d+16+16+8+8, d, batch_first=True)
        self.drop = nn.Dropout(pdrop)
        self.out = nn.Linear(d, n_items+1)

    def forward(self, item_seq, comp, cat_multi_bt, distbin, hour):
        x = self.item_emb(item_seq)
        B,T = item_seq.size()
        comp_e = self.comp_emb(comp).unsqueeze(1).expand(B,T,-1)
        # cat_multi_bt: (B,T,n_cat) 0/1 → 임베 평균
        Nc = cat_multi_bt.size(-1)
        cat_idx = cat_multi_bt.nonzero(as_tuple=False)           # (M,3) = (b,t,k)
        cat_e = torch.zeros((B,T,16), device=item_seq.device)
        if len(cat_idx)>0:
            emb = self.cat_emb(cat_idx[:,2])                     # (M,16)
            bt = cat_idx[:,0]*T + cat_idx[:,1]
            cat_e = cat_e.view(B*T,16)
            cat_e.index_add_(0, bt, emb)
            cat_e = cat_e.view(B,T,16)
            counts = cat_multi_bt.sum(-1, keepdim=True).clamp(min=1.0)
            cat_e = cat_e / counts
        dist_e = self.dist_emb(distbin)
        hour_e = self.hour_emb(hour)
        h_in = torch.cat([x, comp_e, cat_e, dist_e, hour_e], dim=-1)
        h,_ = self.gru(self.drop(h_in))
        h = self.drop(h)
        return self.out(h)                                       # (B,T,V)

# ==== 로더 ====
def load_artifacts(model_path, places_csv, device):
    ckpt = torch.load(model_path, map_location="cpu")
    pid2idx = ckpt["pid2idx"]; idx2pid = ckpt["idx2pid"]
    n_items = len(pid2idx)
    model = ContextGRURec(n_items=n_items)
    model.load_state_dict({k:torch.tensor(v) if isinstance(v,np.ndarray) else v for k,v in ckpt["state_dict"].items()})
    model.to(device).eval()
    places = pd.read_csv(places_csv, encoding="utf-8")
    places = places[places["place_id"].isin(pid2idx.keys())].copy()
    places.set_index("place_id", inplace=True)
    return model, pid2idx, idx2pid, places

# ==== 컨텍스트 텐서 생성 ====
def build_context_tensors(seq_pid, companion, trip_cats, start_lat, start_lng, places, pid2idx, hour_last=10, device="cpu"):
    # item 인덱스 시퀀스
    seq_idx = [pid2idx[p] for p in seq_pid if p in pid2idx]
    if not seq_idx: seq_idx = [0]
    items = torch.tensor(seq_idx, dtype=torch.long, device=device)[None,:]     # (1,T)
    # companion
    comp = torch.tensor([COMP_IDX.get(companion,0)], dtype=torch.long, device=device)
    # cats → (1,T,n_cat)
    cats = np.zeros((1, items.size(1), len(CATS)), dtype=np.float32)
    for c in trip_cats:
        if c in CAT_IDX: cats[0,:,CAT_IDX[c]] = 1.0
    cats_bt = torch.tensor(cats, dtype=torch.float32, device=device)
    # distbin/hour → 마지막 스텝만 사용해 브로드캐스트
    distbins=[]
    for p in seq_pid:
        if p in places.index:
            r = places.loc[p]
            d = haversine(start_lat, start_lng, float(r["lat"]), float(r["lng"]))
            distbins.append(bin_distance_km(d))
        else:
            distbins.append(0)
    if len(distbins)==0: distbins=[0]
    dist = torch.tensor(distbins, dtype=torch.long, device=device)[None,:]
    hour = torch.full((1, items.size(1)), int(hour_last), dtype=torch.long, device=device)
    return items, comp, cats_bt, dist, hour
    
def within_km(lat1,lng1, lat2,lng2, max_km):
    # 하버사인은 기존 함수(haversine) 재사용해도 됨
    from math import radians, sin, cos, asin, sqrt
    R=6371.0
    p1=radians(lat1); p2=radians(lat2)
    dlat=p2-p1; dlng=radians(lng2-lng1)
    a=sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlng/2)**2
    return (2*R*asin(sqrt(a))) <= max_km

def candidate_by_radius_dual(places, start_lat, start_lng, last_lat, last_lng,
                             r_start_km, r_step_km):
    # 시작점 반경 AND 직전 지점 반경 모두 통과한 후보만 리턴
    latv = places["lat"].values; lngv = places["lng"].values
    from numpy import vectorize
    vfun = vectorize(within_km)
    m1 = vfun(start_lat, start_lng, latv, lngv, r_start_km)
    m2 = vfun(last_lat,  last_lng,  latv, lngv, r_step_km)
    mask = m1 & m2
    idx = mask.nonzero()[0]
    return places.iloc[idx].index.to_numpy()
# ==== 반경 후보 및 점수화 ====
def candidate_by_radius(places, lat, lng, r_km):
    d = haversine(lat, lng, places["lat"].values, places["lng"].values)
    cand = places.iloc[(d<=r_km).nonzero()[0]]
    return cand.index.to_numpy(), d[d<=r_km]
def km_between(a_lat,a_lng,b_lat,b_lng):
    R=6371.0
    p1=radians(a_lat); p2=radians(b_lat)
    dlat=p2-p1; dlng=radians(b_lng-a_lng)
    return 2*R*asin((sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlng/2)**2)**0.5)
# 후처리: 완성된 시퀀스에서 과도한 점프 제거/치환
def postprocess_sequence(seq_pid, places, start_lat, start_lng,
                         max_from_start_km=5.0, max_step_km=3.0):
    if not seq_pid: return seq_pid
    cleaned=[seq_pid[0]]
    for p in seq_pid[1:]:
        plat, plng = float(places.loc[p,"lat"]), float(places.loc[p,"lng"])
        ok_start = km_between(start_lat,start_lng, plat,plng) <= max_from_start_km
        prev_lat, prev_lng = float(places.loc[cleaned[-1],"lat"]), float(places.loc[cleaned[-1],"lng"])
        ok_step  = km_between(prev_lat,prev_lng, plat,plng)   <= max_step_km
        if ok_start and ok_step:
            cleaned.append(p)
    return cleaned

@torch.no_grad()
def recommend_next(model, seq_pid, companion, trip_cats, start_lat, start_lng,
                   places, pid2idx, radius_km=5.0, step_radius_km=15.0,
                   topk=10, device="cpu", hour_last=10):

    items, comp, cats_bt, dist, hour = build_context_tensors(
        seq_pid, companion, trip_cats, start_lat, start_lng, places, pid2idx, hour_last, device
    )
    logits = model(items, comp, cats_bt, dist, hour)         # (1,T,V)
    last = logits[:, -1, :].squeeze(0)                       # (V,)

    # 직전 위치
    last_lat = places.loc[seq_pid[-1], "lat"] if seq_pid else start_lat
    last_lng = places.loc[seq_pid[-1], "lng"] if seq_pid else start_lng

    # 반경 후보 산출(시작점 AND 직전지점)
    cand_pids = candidate_by_radius_dual(
        places, start_lat, start_lng, last_lat, last_lng,
        r_start_km=radius_km, r_step_km=step_radius_km
    )
    cand_idx = torch.tensor([pid2idx[p] for p in cand_pids if p in pid2idx],
                            device=device, dtype=torch.long)
    if cand_idx.numel() == 0:
        return []  # 후보 없음

    # ✅ 마스크 수정: 기본 False → 후보만 True
    mask = torch.zeros_like(last, dtype=torch.bool)
    mask[cand_idx] = True

    # 방문지 제외
    for p in set(seq_pid):
        if p in pid2idx:
            mask[pid2idx[p]] = False

    # 마스크 적용
    masked = torch.full_like(last, -1e9)
    masked[mask] = last[mask]

    k = int(min(topk, int(mask.sum().item())))
    if k <= 0:
        return []
    vals, idx = torch.topk(masked, k=k)
    idx2pid = {v:k for k,v in pid2idx.items()}
    return [idx2pid[i.item()] for i in idx]

def build_itinerary(model, L, start_lat, start_lng, companion, trip_cats,
                    required_ids, places, pid2idx, radius_km=5, step_radius_km=10.0, device="cpu"):
    seq = []
    req = [int(x) for x in required_ids if int(x) in places.index]
    # 첫 스텝: required가 있으면 하나부터
    if req:
        seq.append(req.pop(0))
    else:
        # 시작점에서 가장 가까운 첫 후보
        cand_pids, d = candidate_by_radius(places, start_lat, start_lng, radius_km)
        if len(cand_pids)==0: return []
        seq.append(int(cand_pids[np.argmin(d)]))

    for t in range(1, L):
        # 슬롯에 required 주입
        if req and t in {max(1,L//3), max(2,2*L//3)}:
            if req[0] not in seq: seq.append(req.pop(0)); continue
            else: req.pop(0)
        # 모델 추천
        recs = recommend_next(model, seq, companion, trip_cats, start_lat, start_lng,
                              places, pid2idx, radius_km=radius_km, step_radius_km=step_radius_km, topk=10, device=device, hour_last=10+t)
        if len(recs) == 0:
            return seq
        pick = next((p for p in recs if p not in seq), (recs[0] if recs else seq[-1]))
        seq.append(int(pick))
    # 남은 required 치환
    for rid in req:
        for i in range(1, len(seq)-1):
            if seq[i] not in required_ids:
                seq[i] = rid; break
    return seq

async def get_next_poi_list(model, places, start_lat, start_lng, companion, cats, required, length = 6, radius_km=40.0, step_radius_km=20.0, cpu="store_true"):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model, pid2idx, idx2pid, places = load_artifacts(model, places, device)

    seq = build_itinerary(model, L=length,
                        start_lat=start_lat, start_lng=start_lng,
                        companion=companion, trip_cats=cats,
                        required_ids=required, places=places, pid2idx=pid2idx,
                        radius_km=radius_km, step_radius_km=step_radius_km, device=device)
    # seq = postprocess_sequence(seq, places, start_lat, start_lng,
    #                        max_from_start_km=radius_km,
    #                        max_step_km=step_radius_km)
    #place_id 리스트 리턴
    return [idx2pid[i] for i in seq]

# ==== CLI ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--places", default="places_enriched.csv")
    ap.add_argument("--start_lat", type=float, default=37.5665)
    ap.add_argument("--start_lng", type=float, default=126.9780)
    ap.add_argument("--companion", default="친구와", choices=COMPANIONS)
    ap.add_argument("--cats", default="['관광','먹방']")
    ap.add_argument("--required", default="[]")   # 예: "[101,205]"
    ap.add_argument("--length", type=int, default=6)
    ap.add_argument("--radius_km", type=float, default=5.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model, pid2idx, idx2pid, places = load_artifacts(args.model, args.places, device)

    # 입력 파싱
    trip_cats = parse_cats(args.cats)
    required_ids = [int(x) for x in parse_cats(args.required)] if args.required.strip()!="[]" else []

    # 여정 생성
    seq = build_itinerary(model, L=args.length,
                          start_lat=args.start_lat, start_lng=args.start_lng,
                          companion=args.companion, trip_cats=trip_cats,
                          required_ids=required_ids, places=places, pid2idx=pid2idx,
                          radius_km=args.radius_km, device=device)

    # 결과 출력
    out = pd.DataFrame({
        "order": list(range(1,len(seq)+1)),
        "place_id": seq,
        "place": [places.loc[p,"place"] if p in places.index else "" for p in seq],
        "lat": [places.loc[p,"lat"] if p in places.index else None for p in seq],
        "lng": [places.loc[p,"lng"] if p in places.index else None for p in seq],
        "place_type": [places.loc[p,"place_type"] if p in places.index else "" for p in seq],
        "category": [places.loc[p,"category"] if p in places.index else "" for p in seq],
    })
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
