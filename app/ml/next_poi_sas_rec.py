# filename: recommend_next_poi.py
import argparse
from pathlib import Path
import pickle
from typing import List, Iterable, Optional, Tuple, Set, Dict, Any

import math
import pandas as pd
import torch
import torch.nn as nn


# -------------------------
# 공통: 매핑/거리/파서
# -------------------------
def load_mappings(mappings_dir: str = "models") -> tuple[dict, dict]:
    mdir = Path(mappings_dir)
    with open(mdir / "place2idx.pkl", "rb") as f:
        place2idx = pickle.load(f)
    with open(mdir / "idx2place.pkl", "rb") as f:
        idx2place = pickle.load(f)
    return place2idx, idx2place

def to_index_sequence(places: Iterable[int], place2idx: dict, max_len: int = 20) -> torch.Tensor:
    seq_idx = [place2idx[p] for p in places if p in place2idx]
    seq_idx = seq_idx[-max_len:]
    pad = [0] * (max_len - len(seq_idx))
    return torch.tensor(pad + seq_idx, dtype=torch.long).unsqueeze(0)  # (1, L)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def split_multi(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    return [t.strip() for p in s.split("|") for t in p.split(",") if t.strip()]

def norm_tokens(xs: Iterable[str]) -> Set[str]:
    return {x.strip().lower() for x in xs if x and str(x).strip()}

class SASRec(nn.Module):
    def __init__(self, num_items, hidden_size=128, max_len=20, num_heads=2, num_layers=2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.encoder_layers = nn.TransformerEncoder(  # <-- 이름과 구조 학습 코드와 동일
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1
                # batch_first 파라미터 없이 학습과 동일
            ),
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, num_items)

    def forward(self, seq):  # (B, L)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)  # (1, L)
        x = self.item_emb(seq) + self.pos_emb(positions)  # (B, L, D)
        x = self.layernorm(x)
        x = self.encoder_layers(x)  # 학습 코드와 동일 호출
        x = self.dropout(x)
        last_hidden = x[:, -1, :]   # (B, D)
        logits = self.out(last_hidden)  # (B, num_items)
        return logits
# -------------------------
# 후보 필터링 규칙
# -------------------------
def build_candidates(
    places_df: pd.DataFrame,
    place2idx: dict,
    fixed_place_ids: List[int],
    start_lat: float,
    start_lng: float,
    radius_km: float,
    step_radius_km: float,
    companion: str,
    themes: List[str],
) -> Set[int]:
    need_cols = {"place_id", "lat", "lng"}
    missing = [c for c in need_cols if c not in places_df.columns]
    if missing:
        raise ValueError(f"places.csv 누락 컬럼: {missing}  (필수: {sorted(need_cols)})")

    # 1) 시작 반경 필터
    within_radius: Set[int] = set()
    for r in places_df.itertuples(index=False):
        pid = int(getattr(r, "place_id"))
        if pid not in place2idx:
            continue
        d = haversine_km(start_lat, start_lng, float(getattr(r, "lat")), float(getattr(r, "lng")))
        if d <= radius_km:
            within_radius.add(pid)

    # 2) step 반경: 마지막 고정 장소 기준
    if fixed_place_ids:
        last_pid = fixed_place_ids[-1]
        last_row = places_df.loc[places_df["place_id"] == last_pid]
        if not last_row.empty:
            last_lat = float(last_row.iloc[0]["lat"])
            last_lng = float(last_row.iloc[0]["lng"])
            near_last: Set[int] = set()
            for r in places_df.itertuples(index=False):
                pid = int(getattr(r, "place_id"))
                if pid not in place2idx:
                    continue
                d = haversine_km(last_lat, last_lng, float(getattr(r, "lat")), float(getattr(r, "lng")))
                if d <= step_radius_km:
                    near_last.add(pid)
            within_radius = within_radius & near_last if near_last else within_radius

    # 3) 콘텐츠 선호 필터(옵션): theme/companion 컬럼이 있으면 강화
    user_themes = norm_tokens(themes)
    user_comp = str(companion or "").strip().lower()

    def theme_match(row) -> bool:
        # 사용할 컬럼 후보: themes, theme, category
        for col in ["themes", "theme", "category"]:
            if col in places_df.columns:
                item = norm_tokens(split_multi(str(row[col])))
                if user_themes and item and not (user_themes & item):
                    return False
        return True

    def companion_match(row) -> bool:
        # 사용할 컬럼 후보: companions, relation, companion
        for col in ["companions", "relation", "companion"]:
            if col in places_df.columns:
                item = norm_tokens(split_multi(str(row[col])))
                if user_comp and item and user_comp not in item:
                    return False
        return True

    filtered: Set[int] = set()
    for _, row in places_df.loc[places_df["place_id"].isin(list(within_radius))].iterrows():
        if theme_match(row) and companion_match(row):
            filtered.add(int(row["place_id"]))

    # 최소 보장: 너무 적으면 반경만 적용
    if len(filtered) < 5:
        filtered = within_radius

    # 고정 장소는 제외
    filtered -= set(int(x) for x in fixed_place_ids)
    return filtered

# -------------------------
# 재순위 규칙(거리+콘텐츠 보정)
# -------------------------
def rerank(
    base_scores: Dict[int, float],
    places_df: pd.DataFrame,
    start_lat: float,
    start_lng: float,
    themes: List[str],
    alpha_dist: float = 0.03,   # 거리 보정 강도(작게)
    beta_theme: float = 0.10,   # 테마 일치 보정
) -> List[Tuple[int, float, Dict[str, Any]]]:
    user_themes = norm_tokens(themes)
    rows = []
    for pid, s in base_scores.items():
        row = places_df.loc[places_df["place_id"] == pid]
        if row.empty:
            rows.append((pid, s, {"dist_km": None, "theme_hit": 0}))
            continue
        lat = float(row.iloc[0]["lat"])
        lng = float(row.iloc[0]["lng"])
        d = haversine_km(start_lat, start_lng, lat, lng)
        # 테마 히트 계산
        theme_hit = 0
        for col in ["themes", "theme", "category"]:
            if col in row.columns:
                item = norm_tokens(split_multi(str(row.iloc[0][col])))
                if user_themes and (user_themes & item):
                    theme_hit = 1
                    break
        score = s + beta_theme * theme_hit - alpha_dist * d
        rows.append((pid, score, {"dist_km": d, "theme_hit": theme_hit}))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

# -------------------------
# 메인 추천 함수
# -------------------------
@torch.no_grad()
def get_next_poi_list(
    model, 
    places, 
    start_lat, 
    start_lng, 
    companion, 
    cats, 
    required, 
    length = 6, 
    radius_km=40.0, 
    step_radius_km=20.0, 
    cpu="store_true"
) -> List[Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    places_df = pd.read_csv(places)

    all_place_ids = places_df["place_id"].unique().tolist()
    place2idx = {pid: i+1 for i, pid in enumerate(all_place_ids)}
    idx2place = {i+1: pid for i, pid in enumerate(all_place_ids)}
    num_items = max(idx2place.keys()) + 1

    model = SASRec(num_items=num_items, hidden_size=128, max_len=length, num_heads=2, num_layers=2).to(device)
    state = torch.load(model, map_location=device)
    model.load_state_dict(state)
    model.eval()


    # 후보 집합
    candidates = build_candidates(
        places_df=places_df,
        place2idx=place2idx,
        fixed_place_ids=required,
        start_lat=start_lat,
        start_lng=start_lng,
        radius_km=radius_km,
        step_radius_km=step_radius_km,
        companion=companion,
        themes=cats,
    )

    # --- 디버그: 후보 수 집계 ---
    def dbg(msg): 
        print(f"[DEBUG] {msg}")

    dbg(f"fixed={len(required)}, radius_km={radius_km}, step_radius_km={step_radius_km}")
    dbg(f"place2idx size={len(place2idx)}, places.csv rows={len(places_df)}")
    dbg(f"initial candidates={len(candidates)}")

    # --- 점진적 완화 ---
    if not candidates:
        dbg("no candidates -> drop theme/companion + step constraint")
        # 1) 반경만 적용
        within = set()
        for r in places_df.itertuples(index=False):
            pid = int(getattr(r, "place_id"))
            if pid not in place2idx:
                continue
            d = haversine_km(start_lat, start_lng, float(getattr(r, "lat")), float(getattr(r, "lng")))
            if d <= radius_km:
                within.add(pid)
        within -= set(int(x) for x in required)
        candidates = within
        dbg(f"radius-only candidates={len(candidates)}")

    if not candidates:
        dbg("still empty -> expand radius x2")
        expand = radius_km * 2.0
        for r in places_df.itertuples(index=False):
            pid = int(getattr(r, "place_id"))
            if pid not in place2idx:
                continue
            d = haversine_km(start_lat, start_lng, float(getattr(r, "lat")), float(getattr(r, "lng")))
            if d <= expand:
                candidates.add(pid)
        candidates -= set(int(x) for x in required)
        dbg(f"expanded candidates={len(candidates)}")

    if not candidates:
        dbg("still empty -> force top-pop from mapping intersection")
        candidates = {pid for pid in places_df["place_id"].tolist() if pid in place2idx}
        candidates -= set(int(x) for x in required)
        dbg(f"fallback candidates={len(candidates)}")

    if not candidates:
        dbg("no candidates at all -> return []")
        return []

    # 입력 시퀀스
    seq = to_index_sequence(required, place2idx, max_len=length).to(device)

    # 로짓
    logits = model(seq).squeeze(0)  # (num_items,)
    logits[0] = float("-inf")

    # 방문 제외
    for p in required:
        if p in place2idx:
            logits[place2idx[p]] = float("-inf")

    # 후보 이외 -inf
    cand_idx = {place2idx[p] for p in candidates if p in place2idx}
    all_idx = set(range(len(logits)))
    for j in all_idx - cand_idx:
        logits[j] = float("-inf")

    # 상위 넉넉히 뽑기 후 재순위
    k0 = min(max(length * 5, length), int(torch.isfinite(logits).sum().item()))
    if k0 <= 0:
        return []

    top_scores, top_idx = torch.topk(logits, k=k0)
    base_scores = {idx2place[i]: float(s) for i, s in zip(top_idx.tolist(), top_scores.tolist()) if i in idx2place}

    reranked = rerank(
        base_scores=base_scores,
        places_df=places_df,
        start_lat=start_lat,
        start_lng=start_lng,
        themes=cats,
        alpha_dist=0.03,
        beta_theme=0.10,
    )

    out = []
    for pid, score, meta in reranked[:length]:
        out.append({
            "place_id": int(pid),
            "score": float(score),
            "distance_from_start_km": meta.get("dist_km"),
            "theme_hit": int(meta.get("theme_hit", 0)),
        })
    return out

# -------------------------
# CLI
# -------------------------
def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]

def parse_str_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Next-POI recommendation with context filters")
    ap.add_argument("--model", type=str, default="sasrec_context_model.pt")
    ap.add_argument("--places_csv", type=str, default="places.csv")
    ap.add_argument("--fixed", type=str, default="", help="comma-separated place_ids already planned")
    ap.add_argument("--start_lat", type=float, required=True)
    ap.add_argument("--start_lng", type=float, required=True)
    ap.add_argument("--companion", type=str, default="")
    ap.add_argument("--themes", type=str, default="", help="comma-separated themes")
    ap.add_argument("--length", type=int, default=10)
    ap.add_argument("--radius_km", type=float, default=20.0)
    ap.add_argument("--step_radius_km", type=float, default=5.0)
    ap.add_argument("--max_len", type=int, default=20)
    args = ap.parse_args()

    res = get_next_poi_list(
        model_path=args.model,
        places_csv=args.places_csv,
        fixed_place_ids=parse_int_list(args.fixed),
        start_lat=args.start_lat,
        start_lng=args.start_lng,
        companion=args.companion,
        themes=parse_str_list(args.themes),
        length=args.length,
        radius_km=args.radius_km,
        step_radius_km=args.step_radius_km,
        max_len=args.max_len,
    )
    for r in res:
        print(f"{r['place_id']},{r['score']:.6f},{r['distance_from_start_km']:.3f},{r['theme_hit']}")

if __name__ == "__main__":
    main()
