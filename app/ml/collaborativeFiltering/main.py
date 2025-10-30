"""
협업 필터링(Collaborative Filtering) 중에서 모델베이스 행렬분해(Matrix Factorization) 계열
ALS 기반 추천 시스템 (user_id → place_id)
데이터·행렬 형태 요약
train_user_items, test_user_items: CSR, shape = (n_users, n_items), float32.
train_implicit, test_implicit: 이진 CSR, 동일 shape.
매핑:
uid2idx: user_id → [0..n_users)
pid2idx: place_id → [0..n_items)
역매핑 포함.
평가 지표 해석
precision@K: 상위 K 중 정답 비율.
map@K: 평균 정밀도의 상위 K 절단 평균.
ndcg@K: 순위 민감 정규화 누적 이득.
auc@K(있을 때): 상위 K 구간 내 분리능 지표.
예외·엣지 케이스
CSV에 필수 컬럼 누락 → ValueError.
preprocess 이전에 build_mappings 등 호출 → RuntimeError.
split_data 이후 특정 유저가 train/test에만 존재할 수 있음. 행렬 차원은 전체 매핑 기준이라 일관성 유지.
추천 시 user_id 미존재 → ValueError.
시간·공간 복잡도 개괄
매핑 생성: O(U + I).
CSR 생성: O(nnz).
학습: 반복당 대략 O((U + I)·K³)에 가까운 선형시스템 다수 해법 + 희소곱 비용. 실제는 구현 최적화와 스레드 병렬화에 좌우.
추천 단일 유저: O(I·K) 근사. 라이브러리 내부 최적화 적용.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
import pickle

# 평가 함수 import (버전별 AUC 처리)
try:
    from implicit.evaluation import (
        precision_at_k,
        mean_average_precision_at_k,
        ndcg_at_k,
        AUC_at_k,
    )
    HAS_AUC = True
except ImportError:
    from implicit.evaluation import (
        precision_at_k,
        mean_average_precision_at_k,
        ndcg_at_k,
    )
    HAS_AUC = False


class ALSRecommender:
    """ALS 기반 추천 시스템 (user_id → place_id)
    - 입력 데이터 컬럼 요구사항: 'user_id', 'place_id', 'rating'
    """

    def __init__(self,
                 factors: int = 64,
                 regularization: float = 0.08,
                 iterations: int = 20,
                 use_gpu: bool = False):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu

        self.df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self.uid2idx: dict[str, int] | None = None
        self.pid2idx: dict[str | int, int] | None = None
        self.idx2uid: dict[int, str] | None = None
        self.idx2pid: dict[int, str | int] | None = None

        self.train_user_items: csr_matrix | None = None
        self.test_user_items: csr_matrix | None = None
        self.train_implicit: csr_matrix | None = None
        self.test_implicit: csr_matrix | None = None

        self.model: AlternatingLeastSquares | None = None

    # ---------- 데이터 로드/전처리 ----------
    def load_data(self, csv_path: str = "good_data.csv") -> None:
        """
        입력: CSV 경로. 필요한 컬럼: user_id, place_id, rating.
        처리: CSV 로드 → 필수 컬럼 확인 → 서브셋·결측 제거.
        상태 변화: self.df 설정.
        예외: 누락 컬럼 시 ValueError.
        출력: 없음.
        """
        self.df = pd.read_csv(csv_path)
        required = {"user_id", "place_id", "rating"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")
        self.df = self.df[["user_id", "place_id", "rating"]].dropna()
        print(f"✓ 데이터 로드: {len(self.df)} rows")

    def preprocess(self) -> None:
        """
        전제: self.df 존재.
        처리: 인덱스 리셋. 추가 전처리 없음.
        상태 변화: self.df 덮어씀.
        예외: 미호출 시 RuntimeError.
        출력: 없음.
        """
        if self.df is None:
            raise RuntimeError("먼저 load_data()를 호출하세요.")
        # user_id, place_id, rating만 사용
        self.df = self.df.reset_index(drop=True)
        print(f"✓ 전처리 완료: 행수={len(self.df)}")

    def build_mappings(self) -> None:
        """
        전제: self.df 존재.
        처리: 매핑 생성.
            - uid2idx: 고유 user_id 정렬 후 0..U-1 매핑.
            - pid2idx: 고유 place_id 정렬 후 0..I-1 매핑.
            - idx2uid: uid2idx 역매핑.
            - idx2pid: pid2idx 역매핑.
        상태 변화: self.uid2idx, self.pid2idx, self.idx2uid, self.idx2pid 설정.
        예외: 미호출 시 RuntimeError.
        출력: 없음.
        """
        if self.df is None:
            raise RuntimeError("먼저 load_data()/preprocess()를 호출하세요.")
        self.uid2idx = {u: i for i, u in enumerate(sorted(self.df["user_id"].unique()))}
        self.pid2idx = {p: i for i, p in enumerate(sorted(self.df["place_id"].unique()))}
        self.idx2uid = {i: u for u, i in self.uid2idx.items()}
        self.idx2pid = {i: p for p, i in self.pid2idx.items()}
        print(f"✓ 매핑 생성: users={len(self.uid2idx)}, items={len(self.pid2idx)}")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        전제: self.df 존재.
        처리: rating 기준 층화 분할. 학습/평가 분리
        상태 변화: self.train_df, self.test_df 설정.
        예외: 미호출 시 RuntimeError.
        출력: 없음.
        """
        if self.df is None:
            raise RuntimeError("먼저 load_data()/preprocess()를 호출하세요.")
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=test_size, stratify=self.df["rating"], random_state=random_state
        )
        print(f"✓ 데이터 분할: train={len(self.train_df)}, test={len(self.test_df)}")

    # ---------- 행렬 유틸 ----------
    def _ensure_csr32(self, m: csr_matrix) -> csr_matrix:
        """
        입력: 임의 CSR
        처리: CSR로 강제, float32, 인덱스 정렬.
        출력: csr_matrix. (정규화된 CSR)
        복잡도: O(nnz)
        """
        m = m.tocsr().astype(np.float32)
        m.sort_indices()
        return m

    def _build_user_item(self, source_df: pd.DataFrame, rating_col: str = "rating") -> csr_matrix:
        """
        전제: source_df 존재, uid2idx, pid2idx 설정.
        처리: rows=user_id→uid2idx, cols=place_id→pid2idx, data=rating
        상태 변화: x
        예외: 미호출 시 RuntimeError.
        출력: csr_matrix. (n_users, n_items)의 CSR
        복잡도: O(nnz)
        """
        if self.uid2idx is None or self.pid2idx is None:
            raise RuntimeError("먼저 build_mappings()를 호출하세요.")
        rows = source_df["user_id"].map(self.uid2idx).astype(np.int32).to_numpy()
        cols = source_df["place_id"].map(self.pid2idx).astype(np.int32).to_numpy()
        data = source_df[rating_col].astype(np.float32).to_numpy()
        m = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.uid2idx), len(self.pid2idx)),
            dtype=np.float32,
        )
        m.sort_indices()
        return m

    def _binarize_geq3(self, m: csr_matrix) -> csr_matrix:
        """
        입력: CSR
        처리: data>=3를 1.0, 그 외 0으로 이진화. 0 제거. 인덱스 정렬
        출력: 이진 CSR
        비고: 임계값 3 고정
        """
        m = m.copy().tocsr()
        mask = m.data >= 3.0
        m.data[:] = mask.astype(np.float32)
        m.eliminate_zeros()
        m.sort_indices()
        return m

    def build_matrices(self) -> None:
        """
        전제: train_df, test_df 존재.
        처리:
            - 학습/테스트에서 유저–아이템 행렬 생성
            - 각각 이진화하여 train_implicit, test_implicit 생성
        상태 변화: 네 개 CSR 속성 설정.
        예외: 미호출 시 RuntimeError.
        """
        if self.train_df is None or self.test_df is None:
            raise RuntimeError("먼저 split_data()를 호출하세요.")
        self.train_user_items = self._ensure_csr32(self._build_user_item(self.train_df))
        self.test_user_items = self._ensure_csr32(self._build_user_item(self.test_df))
        self.train_implicit = self._binarize_geq3(self.train_user_items)
        self.test_implicit = self._binarize_geq3(self.test_user_items)
        print("✓ 행렬 생성 완료 (train/test implicit)")

    # ---------- 학습/평가 ----------
    def train(self, use_bm25: bool = False) -> None:
        """
        전제: train_implicit 존재.
        처리:
            - train_for_fit = train_implicit 또는 bm25_weight(train_implicit) 후 float32.
            - AlternatingLeastSquares 인스턴스 생성.
            - model.fit(train_for_fit).
        상태 변화: self.model 학습 완료.
        예외: 미호출 시 RuntimeError.
        입력 하이퍼파라미터 영향:
            - factors: 잠재 차원
            - regularization: L2 규제
            - iterations: ALS 반복 수
            - use_gpu: GPU 사용 여부
        """
        if self.train_implicit is None:
            raise RuntimeError("먼저 build_matrices()를 호출하세요.")
        train_for_fit = self._ensure_csr32(bm25_weight(self.train_implicit)) if use_bm25 else self.train_implicit
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
        )
        # implicit는 recommend 호출 시 user_items를 넘기므로 여기서는 그대로 fit 가능
        self.model.fit(train_for_fit)
        print("✓ ALS 학습 완료")

    def evaluate(self, K: int = 10) -> dict:
        """
        전제: model, train_implicit, test_implicit 존재.
        처리: precision@K, map@K, ndcg@K, 옵션으로 auc@K.
        출력: {metric_name: float} 딕셔너리.
        상태 변화: 없음.
        예외: 내부 지표 계산 중 예외를 잡아 메시지 출력 후 해당 값 미기록 또는 키 다름(코드 그대로).
        """
        if self.model is None or self.train_implicit is None or self.test_implicit is None:
            raise RuntimeError("먼저 train()을 완료하세요.")
        results = {}
        try:
            results[f"precision@{K}"] = float(
                precision_at_k(self.model, self.train_implicit, self.test_implicit, K=int(K), show_progress=False)
            )
        except Exception as e:
            print(f"Error In precision_at_k: {e}")
        try:
            results[f"map@{K}"] = float(
                mean_average_precision_at_k(self.model, self.train_implicit, self.test_implicit, K=int(K), show_progress=False)
            )
        except Exception as e:
            print(f"Error In mean_average_precision_at_k: {e}")
        try:
            results[f"ndcg@{K}"] = float(
                ndcg_at_k(self.model, self.train_implicit, self.test_implicit, K=int(K), show_progress=False)
            )
        except Exception as e:
            print(f"Error In ndcg_at_k: {e}")
        if HAS_AUC:
            try:
                results[f"auc@{K}"] = float(AUC_at_k(self.model, self.train_implicit, self.test_implicit, K=int(K), show_progress=False))
            except TypeError:
                results["auc"] = float(AUC_at_k(self.model, self.train_implicit, self.test_implicit))
            except Exception as e:
                print(f"Error In AUC_at_k: {e}")
        print("📊 평가:", results)
        return results

    # ---------- 추천 ----------
    def recommend(self,
                  user_id,
                  N: int = 10,
                  filter_already_liked_items: bool = False) -> list[tuple[str | int, float]]:
        """
        전제: model, train_implicit, uid2idx, idx2pid 존재.
        처리:
           - user_id를 내부 인덱스로 변환.
           - model.recommend(userid=uid, user_items=train_implicit, N, filter_already_liked_items=...).
           - 아이템 인덱스를 place_id로 역매핑.
        출력: [(place_id, score), ...] 길이 N 리스트.
        상태 변화: 없음.
        예외: 미등록 user_id는 ValueError.
        점수 의미: 사용자·아이템 잠재벡터 내적 기반 추천 점수.
        """
        if self.model is None or self.train_implicit is None:
            raise RuntimeError("먼저 train()을 완료하세요.")
        if self.uid2idx is None or self.idx2pid is None:
            raise RuntimeError("먼저 build_mappings()을 완료하세요.")
        if user_id not in self.uid2idx:
            raise ValueError(f"등록되지 않은 user_id: {user_id}")
        uid = self.uid2idx[user_id]
        rec_idx, rec_scores = self.model.recommend(
            userid=uid,
            user_items=self.train_implicit,
            N=int(N),
            filter_already_liked_items=filter_already_liked_items,
        )
        return [(self.idx2pid[int(i)], float(s)) for i, s in zip(rec_idx, rec_scores)]

    def recommend_df(self,
                     user_id,
                     N: int = 10,
                     filter_already_liked_items: bool = False) -> pd.DataFrame:
        """
        전제: recommend 가능.
        처리: recommend 호출 → DataFrame 변환 → 소수점 4자리 반올림
        출력: 컬럼 ["place_id","score"]의 pd.DataFrame
        """
        recs = self.recommend(user_id=user_id, N=N, filter_already_liked_items=filter_already_liked_items)
        return pd.DataFrame(recs, columns=["place_id", "score"]).assign(score=lambda d: d["score"].round(4))

    # ---------- 저장/로드 ----------
    def save_model(self, model_path: str = "als_model.pkl") -> bool:
        """
        전제: model, uid2idx, pid2idx 존재.
        처리:
            - 파라미터, user_factors, item_factors, 매핑 4종을 dict로 피클 저장
        출력: 성공 True, 실패 False.
        예외: 전제 불만족 시 RuntimeError. 저장 중 예외는 잡아 메시지 출력 후 False.
        """
        if self.model is None or self.uid2idx is None or self.pid2idx is None:
            raise RuntimeError("모델/매핑 정보가 없습니다. 먼저 train() 및 build_mappings()를 완료하세요.")
        try:
            data = {
                "params": {
                    "factors": self.factors,
                    "regularization": self.regularization,
                    "iterations": self.iterations,
                    "use_gpu": self.use_gpu,
                },
                # implicit ALS는 전체 객체 피클보다 factor 행렬 저장이 안전
                "user_factors": getattr(self.model, "user_factors", None),
                "item_factors": getattr(self.model, "item_factors", None),
                "uid2idx": self.uid2idx,
                "pid2idx": self.pid2idx,
                "idx2uid": self.idx2uid,
                "idx2pid": self.idx2pid,
            }
            with open(model_path, "wb") as f:
                pickle.dump(data, f)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ ALS 모델 저장: {model_path} ({size_mb:.2f} MB)")
            return True
        except Exception as e:
            print(f"✗ ALS 모델 저장 실패: {e}")
            return False

    def load_model(self, model_path: str = "als_model.pkl") -> bool:
        """
        입력: 피클 경로.
        처리:
            - 파일 확인 후 로드.
            - 저장된 파라미터로 모델 인스턴스 재생성.
            - user_factors, item_factors가 있으면 주입.
            - 매핑 4종 복구.
        출력: 성공 True, 실패 False.
        부가 출력: 파일 크기, 유저·아이템 수 표시.
        예외: 로드 중 예외는 잡아 메시지 출력 후 False.
        """
        try:
            if not os.path.exists(model_path):
                print(f"✗ 모델 파일이 없습니다: {model_path}")
                return False
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            params = data.get("params", {})
            self.factors = int(params.get("factors", self.factors))
            self.regularization = float(params.get("regularization", self.regularization))
            self.iterations = int(params.get("iterations", self.iterations))
            self.use_gpu = bool(params.get("use_gpu", self.use_gpu))

            # 매핑 복구
            self.uid2idx = data.get("uid2idx")
            self.pid2idx = data.get("pid2idx")
            self.idx2uid = data.get("idx2uid")
            self.idx2pid = data.get("idx2pid")

            # 모델 인스턴스 생성 후 factor 주입
            self.model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                use_gpu=self.use_gpu,
            )
            uf = data.get("user_factors")
            itf = data.get("item_factors")
            if uf is None or itf is None:
                print("⚠️ 저장된 factor가 없어 재학습이 필요합니다.")
            else:
                self.model.user_factors = uf
                self.model.item_factors = itf

            # 추천이 곧장 가능하도록 빈 사용자-아이템 행렬을 초기화
            # (필터링/가중 목적의 학습 행렬이 없을 경우에도 API 요구사항 충족)
            from scipy.sparse import csr_matrix as _csr
            if self.uid2idx is not None and self.pid2idx is not None:
                self.train_implicit = _csr((len(self.uid2idx), len(self.pid2idx)), dtype=np.float32)

            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ ALS 모델 로드: {model_path} ({size_mb:.2f} MB)")
            print(f"  - users: {len(self.uid2idx) if self.uid2idx else 0}, items: {len(self.pid2idx) if self.pid2idx else 0}")
            return True
        except Exception as e:
            print(f"✗ ALS 모델 로드 실패: {e}")
            return False


def main():
    recommender = ALSRecommender()
    # 1) 저장된 모델 로드 시도
    loaded = recommender.load_model("als_model.pkl")

    # 2) 모델/팩터가 없으면 학습 파이프라인 실행
    needs_training = (
        not loaded or
        recommender.model is None or
        getattr(recommender.model, "user_factors", None) is None or
        getattr(recommender.model, "item_factors", None) is None or
        recommender.uid2idx is None or recommender.pid2idx is None
    )

    if needs_training:
        print("🔄 저장된 모델이 없거나 불완전합니다. 학습을 시작합니다...")
        recommender.load_data("good_data.csv")
        recommender.preprocess()
        recommender.build_mappings()
        recommender.split_data(test_size=0.2)
        recommender.build_matrices()
        recommender.train(use_bm25=False)
        recommender.evaluate(K=10)
        recommender.save_model("als_model.pkl")

    # 3) 예시 추천: 데이터에 존재하는 임의의 user_id 선택
    try:
        example_user = next(iter(recommender.uid2idx.keys())) if recommender.uid2idx else None
        if example_user is None:
            print("추천 에러: 사용자 없음")
        else:
            print(f"🧪 샘플 사용자: {example_user}")
            df_rec = recommender.recommend_df(example_user, N=10)
            print(df_rec.head())
    except Exception as e:
        print(f"추천 에러: {e}")

    # 디버그: 형상/타입 확인이 필요할 때만 사용
    # print("train_implicit dtype:", recommender.train_implicit.dtype,
    #       recommender.train_implicit.data.dtype, recommender.train_implicit.indices.dtype, recommender.train_implicit.indptr.dtype)

    return recommender


if __name__ == "__main__":
    main()