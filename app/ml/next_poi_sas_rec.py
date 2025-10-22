# infer_nextpoi_sasrec.py
# 사용: python infer_nextpoi_sasrec.py
# 준비물:
#   1) 체크포인트: ./out/nextpoi_sasrec.pt   (train_next_poi_sasrec.py로 학습)
#   2) 장소파일:   ./out/places.xlsx 또는 ./out/places_with_ids.csv
# 기능:
#   - prefix 시퀀스+컨텍스트로 Top-K 다음 장소 추천
#   - must_visit(필수방문)·already_visited 마스킹
#   - 테마/지역좌표 기반 후보 축소

import os, math, json, argparse
from ast import literal_eval
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ======= 모델 정의(SASRecNextPOI: 학습 스크립트와 동일) =======
class SASRecNextPOI(nn.Module):
    def __init__(self, n_items:int, ctx_dim:int, d:int=256, heads:int=8, layers:int=3, drop:float=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+1, d, padding_idx=0)
        self.pos_emb  = nn.Embedding(512, d)
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim,256), nn.GELU(), nn.Linear(256,d), nn.LayerNorm(d))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4*d, dropout=drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def encode_query(self, seq_ids, attn_mask, ctx_vec):
        B,L = seq_ids.size()
        pos = torch.arange(L, device=seq_ids.device).unsqueeze(0).expand(B,L)
        x = self.item_emb(seq_ids) + self.pos_emb(pos) + self.ctx_mlp(ctx_vec)[:,None,:]
        causal = torch.triu(torch.ones(L,L, device=seq_ids.device), diagonal=1).bool()
        h = self.encoder(x, mask=causal, src_key_padding_mask=~attn_mask)
        h = self.norm(h)
        last_idx = attn_mask.long().sum(1)-1
        q = self.drop(h[torch.arange(B, device=h.device), last_idx])
        return q  # [B,D]

# ======= 유틸 =======
def parse_cat_cell(x):
    if pd.isna(x): return []
    s=str(x).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            arr = literal_eval(s); return [str(v).strip() for v in arr if str(v).strip()]
        except Exception:
            pass
    return [t.strip() for t in s.split(',') if t.strip()]

def load_places(places_path:str):
    p=str(places_path)
    if p.lower().endswith(".csv"): df=pd.read_csv(p)
    else: df=pd.read_excel(p)
    need={"place","address","lat","lng","place_type","category"}
    miss=need - set(df.columns)
    if miss: raise ValueError(f"places 필수 컬럼 누락: {miss}")
    if "place_id" not in df.columns:
        df=df.copy(); df["place_id"]=np.arange(1,len(df)+1,dtype=int)
    df["place_id"]=df["place_id"].astype(int)
    df["lat"]=pd.to_numeric(df["lat"], errors="coerce")
    df["lng"]=pd.to_numeric(df["lng"], errors="coerce")
    df=df.dropna(subset=["lat","lng"])
    df["categories_list"]=df["category"].apply(parse_cat_cell)
    return df

def hav_km(lat1, lng1, lat2, lng2):
    R=6371.0; p=np.pi/180.0
    dlat=(lat2-lat1)*p; dlng=(lng2-lng1)*p
    a=np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlng/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def load_ckpt_any(ckpt_path:str):
    ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("state_dict","model_state","model_state_dict","weights","params"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd=ckpt[k]; meta=ckpt.get("meta", {}); cfg=ckpt.get("cfg", {})
                return sd, meta, cfg
        # raw state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt, {}, {}
    # nn.Module 저장 케이스
    if hasattr(ckpt, "state_dict"): 
        return ckpt.state_dict(), getattr(ckpt,"meta",{}), {}
    raise ValueError("지원하지 않는 체크포인트 형식")

# ======= 로더 =======
def build_meta_from_ckpt_or_data(meta_ckpt:dict, places:pd.DataFrame):
    # pid2ix
    place_ids = places["place_id"].astype(int).tolist()
    pid2ix = meta_ckpt.get("pid2ix") or {pid:i+1 for i,pid in enumerate(sorted(place_ids))}
    ix2pid = meta_ckpt.get("ix2pid") or {i:k for k,i in pid2ix.items()}
    n_items = meta_ckpt.get("n_items", len(pid2ix))

    # 컨텍스트 정의(학습 스크립트와 동일)
    companions = meta_ckpt.get("companions") or ["혼자서","친구와","연인과","배우자와","부모님과","아이와"]
    themes = meta_ckpt.get("themes") or ["액티비티","자연","바다","산","쇼핑","문화예술","관광","먹방","힐링"]
    regions = meta_ckpt.get("regions") or []  # 좌표 버킷 없으면 빈 리스트 허용
    meta = {
        "n_items": n_items,
        "pid2ix": pid2ix, "ix2pid": ix2pid,
        "companions": companions, "comp2ix": {c:i for i,c in enumerate(companions)},
        "themes": themes, "theme2ix": {t:i for i,t in enumerate(themes)},
        "regions": regions, "reg2ix": {r:i for i,r in enumerate(regions)},
    }
    # log_pop
    lp = meta_ckpt.get("log_pop")
    if lp is None:
        log_pop = np.zeros(n_items+1, dtype=np.float32)
    else:
        lp = np.asarray(lp, dtype=np.float32)
        if lp.shape[0]==n_items: 
            log_pop = np.concatenate([np.zeros(1,np.float32), lp])
        elif lp.shape[0]==n_items+1:
            log_pop = lp
        else:
            log_pop = np.zeros(n_items+1, dtype=np.float32)
    return meta, log_pop

def ctx_vector(meta, region_lat:float, region_lng:float, companions:str, themes:List[str]):
    # 학습 시 region을 버킷 토큰으로 썼다면 그대로 쓰고, 없으면 연속값을 투입하지 않고 0-벡터로 둔다.
    R=len(meta["regions"]); C=len(meta["companions"]); T=len(meta["themes"])
    reg = torch.zeros(R)
    # 가장 가까운 버킷이 있으면 1-hot
    if R>0:
        # regions: "lat,lng" 문자열
        def to_xy(s): 
            a,b = s.split(","); return float(a), float(b)
        coords=np.array([to_xy(s) for s in meta["regions"]], dtype=float)
        dist = hav_km(coords[:,0], coords[:,1], region_lat, region_lng)
        reg[int(np.argmin(dist))]=1.0
    cp = torch.zeros(C); th = torch.zeros(T)
    if companions in meta["comp2ix"]: cp[meta["comp2ix"][companions]]=1.0
    for t in themes:
        if t in meta["theme2ix"]: th[meta["theme2ix"][t]]=1.0
    return torch.cat([reg, cp, th], 0).unsqueeze(0)  # [1, R+C+T]

# ======= 후보 생성 =======
THEME2CAT = {
    "액티비티": ["액티비티","오락","레저","스포츠","VR","테마파크"],
    "자연": ["자연","공원","숲","계곡","정원","호수"],
    "바다": ["바다","해변","서핑","스노클링","항구"],
    "산": ["산","등산","전망대","트레일"],
    "쇼핑": ["쇼핑","시장","상점가","아울렛","백화점"],
    "문화예술": ["전시장","미술관","박물관","공연","도서관"],
    "관광": ["관광지","랜드마크","전통","유적","핫플"],
    "먹방": ["음식점","맛집","노포","카페","야시장"],
    "힐링": ["스파","온천","찜질방","카페","공원"]
}

def build_candidate_pool(places:pd.DataFrame, meta:dict,
                         region_lat:float, region_lng:float,
                         themes:List[str], radius_km:float=50.0,
                         cand_size:int=400,
                         include_pids:Optional[List[int]]=None) -> np.ndarray:
    include_pids = include_pids or []
    cats=set(sum([THEME2CAT.get(t,[]) for t in themes], []))
    lat = places["lat"].to_numpy(); lng = places["lng"].to_numpy()
    d = hav_km(lat, lng, region_lat, region_lng)

    # 1) 반경 필터
    mask = d <= radius_km
    if mask.sum() < cand_size//2:
        # 반경이 너무 작으면 상위 근접으로 확장
        nearest_idx = np.argsort(d)[:max(cand_size, 1000)]
        mask = np.zeros(len(places), dtype=bool); mask[nearest_idx]=True

    # 2) 테마 교집합 가중
    match = places["categories_list"].apply(lambda a: len(set(a)&cats)>0).to_numpy(bool)
    pri = 1.5*match.astype(float) + 1.0  # 테마 일치 가중

    # 3) 거리 가중
    near = 1.0/(1.0 + d.clip(min=0.1))
    score = (pri * near) * mask.astype(float)

    # 포함 보장
    if include_pids:
        inc_ix = np.array([meta["pid2ix"].get(pid,0) for pid in include_pids if pid in meta["pid2ix"]], dtype=int)
        inc_m = np.isin(places["place_id"].to_numpy(), include_pids)
        score[inc_m] = score.max() + 1.0

    idx = np.argsort(-score)[:max(cand_size, 50)]
    # item-idx(ix)로 변환
    pid = places["place_id"].to_numpy()[idx]
    ix = np.array([meta["pid2ix"].get(int(p),0) for p in pid], dtype=int)
    ix = ix[ix>0]
    return ix  # [C]

# ======= 추론 =======
@torch.no_grad()
async def recommend_next(ckpt_path:str,
                   places_path:str,
                   prefix_place_ids:List[int],
                   region_lat:float, region_lng:float,
                   companions:str, themes:List[str],
                   topk:int=10, cand_size:int=400, radius_km:float=10.0,
                   must_visit:Optional[List[int]]=None,
                   already_visited:Optional[List[int]]=None,
                   alpha_pop:float=0.2,
                   center_cut_km:float=20.0
                   ):
    device="cuda" if torch.cuda.is_available() else "cpu"

    # ckpt / meta
    state_dict, meta_ckpt, _ = load_ckpt_any(ckpt_path)
    places = load_places(places_path)
    meta, log_pop = build_meta_from_ckpt_or_data(meta_ckpt, places)

    # 모델 구성(학습 스크립트 구조)
    ctx_dim = len(meta["regions"])+len(meta["companions"])+len(meta["themes"])
    model = SASRecNextPOI(meta["n_items"], ctx_dim, d=256, heads=8, layers=3, drop=0.2)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # 입력 시퀀스
    ids = [meta["pid2ix"].get(int(pid),0) for pid in prefix_place_ids if int(pid) in meta["pid2ix"]]
    if not ids: raise ValueError("prefix_place_ids가 모두 OOV 입니다.")
    seq = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(seq, dtype=torch.bool)
    ctx = ctx_vector(meta, region_lat, region_lng, companions, themes).to(device)

    # 후보
    pool_ix = build_candidate_pool(places, meta, region_lat, region_lng, themes, radius_km, cand_size,
                                   include_pids=must_visit or [])
    pool = torch.tensor(pool_ix, dtype=torch.long, device=device).unsqueeze(0)  # [1,C]

    # 쿼리 인코딩
    q = model.encode_query(seq, attn, ctx)[0]  # [D]

    # 점수(임베딩 내적 + 인기도 prior)
    emb = model.item_emb(pool[0])              # [C,D]
    logits = (emb @ q).float()                 # [C]
    lp = torch.from_numpy(log_pop).to(device)[pool[0]]
    logits = logits + alpha_pop * lp

    # 마스킹
    ban_ix = set()
    for lst in [already_visited or [], prefix_place_ids or []]:
        ban_ix.update([meta["pid2ix"].get(int(pid),0) for pid in lst if int(pid) in meta["pid2ix"]])
    if ban_ix:
        mask = torch.ones_like(logits, dtype=torch.bool)
        pool_list = pool[0].tolist()
        for i,ix in enumerate(pool_list):
            if ix in ban_ix: mask[i]=False
        logits[~mask] = -1e9

    pid_pool = [meta["ix2pid"][int(ix)] for ix in pool[0].tolist()]
    lat_arr = torch.tensor(places.set_index("place_id").loc[pid_pool]["lat"].values, device=device)
    lng_arr = torch.tensor(places.set_index("place_id").loc[pid_pool]["lng"].values, device=device)

    def hav_km_t(a,b,c,d):
        p = math.pi/180.0
        dlat=(c-a)*p; dlng=(d-b)*p
        A=(torch.sin(dlat/2)**2 + torch.cos(a*p)*torch.cos(c*p)*(torch.sin(dlng/2)**2))
        return 2*6371.0*torch.asin(torch.sqrt(torch.clamp(A,0,1)))

    dist_center = hav_km_t(torch.tensor(region_lat,device=device), torch.tensor(region_lng,device=device),
                        lat_arr, lng_arr)
    logits[dist_center > center_cut_km] = -1e9

    # 마지막 방문지 기준 하드 컷(점프 방지)
    if len(prefix_place_ids) > 0:
        last_pid = prefix_place_ids[-1]
        if last_pid in meta["pid2ix"]:
            last = places.set_index("place_id").loc[int(last_pid)]
            dist_last = hav_km_t(torch.tensor(float(last["lat"]),device=device),
                                torch.tensor(float(last["lng"]),device=device),
                                lat_arr, lng_arr)
            logits[dist_last > 15.0] = -1e9   
    # Top-K 반환
    k=min(topk, logits.numel())
    vals, idx = torch.topk(logits, k=k)
    ix_sel = pool[0][idx].tolist()
    pid_sel = [meta["ix2pid"][int(ix)] for ix in ix_sel]
    prob = torch.softmax(vals, dim=0).detach().cpu().numpy().tolist()

    # 보기 좋은 결과 묶기
    places_idx = places.set_index("place_id")
    results=[]
    for pid, p in zip(pid_sel, prob):
        r = places_idx.loc[int(pid)]
        results.append({
            "place_id": int(pid),
            "place": str(r["place"]),
            "address": str(r["address"]),
            "lat": float(r["lat"]), "lng": float(r["lng"]),
            "score": float(p),
            "categories": list(map(str, r["categories_list"]))
        })
    return results

# ======= 예시 실행 =======
if __name__ == "__main__":
    # 예시 파라미터(필요에 따라 수정)
    CKPT = "./out/nextpoi_sasrec.pt"
    PLACES = "./out/places.xlsx"  # 또는 ./out/places_with_ids.csv
    prefix = [101, 202, 303]      # 지금까지 방문한 place_id들
    region_lat, region_lng = 37.5665, 126.9780
    companions = "연인과"
    themes = ["힐링","먹방"]

    recs = recommend_next(
        ckpt_path=CKPT,
        places_path=PLACES,
        prefix_place_ids=prefix,
        region_lat=region_lat, region_lng=region_lng,
        companions=companions, themes=themes,
        topk=10, cand_size=400, radius_km=40.0,
        must_visit=None, already_visited=prefix, alpha_pop=0.2
    )
    # 출력
    for r in recs:
        print(f'{r["place_id"]}\t{r["place"]}\t{r["score"]:.3f}\t{r["categories"][:3]}')
