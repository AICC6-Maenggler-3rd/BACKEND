import pandas as pd
import numpy as np
import ast
import pickle
import os
from scipy.sparse import hstack  # Sparse matrix 결합용
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import warnings
warnings.filterwarnings('ignore')

class TravelRecommendationSystem:
    """여행지 추천 시스템 클래스"""

    def __init__(self):
        self.places_df = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = TfidfVectorizer()
        self.mlb_category = MultiLabelBinarizer()
        self.mlb_suitable = MultiLabelBinarizer()
        self.feature_matrix = None
        # Sparse matrix용 별도 저장 (메모리 최적화)
        self.tfidf_matrix = None
        self.category_matrix = None
        self.suitable_matrix = None
        self.scaler = StandardScaler()
        self.address_similarity_matrix = None  # 지역명 유사도 행렬
        self.address_vectorizer = TfidfVectorizer()  # 지역명 임베딩용

    def load_data(self, csv_path='final_data.csv'):
        """
        CSV 파일에서 데이터 로드
        예상 컬럼:
        - place_id: 장소 ID
        - name: 장소명
        - address: 주소명
        - category: 카테고리
        - suitable: 적합한 구성원 (리스트 또는 콤마 구분)
        - address_la: 위도
        - address_lo: 경도
        - type: 장소 유형
        - count: 장소 개수
        - website: 웹사이트
        - image_url: 이미지 URL
        - insta_nickname: 인스타그램 닉네임
        - open_hour: 오픈 시간
        - close_hour: 클로즈 시간
        - description: 설명
        """
        try:
            self.places_df = pd.read_csv(csv_path)
            print(f"✓ 데이터 로드 완료: {len(self.places_df)}개 장소")
            return True
        except Exception as e:
            print(f"✗ 데이터 로드 실패: {e}")
            return False

    def preprocess_data(self):
        """데이터 전처리"""

        print("\n=== 데이터 전처리 시작 ===")

        # 문자열 형태의 리스트를 실제 리스트로 변환하는 함수
        def parse_string_list(value):
            """문자열 형태의 리스트를 파싱"""
            if pd.isna(value):
                return []
            if isinstance(value, str):
                try:
                    # ast.literal_eval을 사용하여 문자열 형태의 리스트를 실제 리스트로 변환
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        return parsed
                    else:
                        return [parsed]
                except (ValueError, SyntaxError):
                    # 파싱 실패 시 콤마로 분리
                    return [x.strip() for x in value.split(',')]
            if isinstance(value, list):
                return value
            return []

        # 테마와 구성원을 리스트로 변환
        if 'category' in self.places_df.columns:
            self.places_df['category_list'] = self.places_df['category'].apply(parse_string_list)

        if 'suitable' in self.places_df.columns:
            self.places_df['suitable_list'] = self.places_df['suitable'].apply(parse_string_list)
        
        # 설명 컬럼이 없으면 생성
        if 'description' not in self.places_df.columns:
            # categorys 컬럼이 있으면 사용, 없으면 category 사용
            if 'categorys' in self.places_df.columns:
                self.places_df['description'] = (
                    self.places_df['name'] + ' ' +
                    self.places_df['category'].astype(str) + ' ' +
                    self.places_df['categorys'].astype(str)
                )
            else:
                self.places_df['description'] = (
                    self.places_df['name'] + ' ' +
                    self.places_df['category'].astype(str)
                )

        print(f"✓ 전처리 완료")

    def build_features(self):
        """특징 행렬 구성"""

        print("\n=== 특징 행렬 구성 ===")

        #  1. TF-IDF 벡터 (텍스트 설명)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.places_df['description']
        )
        print(f"✓ TF-IDF 행렬: {self.tfidf_matrix.shape} (sparse)")

        # 2. 카테고리 원-핫 인코딩
        self.category_matrix = self.mlb_category.fit_transform(
            self.places_df['category_list']
        )
        print(f"✓ 카테고리 행렬: {self.category_matrix.shape}")
        print(f"  카테고리 종류: {self.mlb_category.classes_.tolist()}")

        # 3. 구성원 원-핫 인코딩
        self.suitable_matrix = self.mlb_suitable.fit_transform(
            self.places_df['suitable_list']
        )
        print(f"✓ 구성원 행렬: {self.suitable_matrix.shape}")
        print(f"  구성원 종류: {self.mlb_suitable.classes_.tolist()}")

        # 4. 지역명 임베딩 및 유사도 행렬 생성
        address_matrix = self.address_vectorizer.fit_transform(
            self.places_df['address']
        )
        self.address_similarity_matrix = cosine_similarity(address_matrix)
        print(f"✓ 지역명 임베딩 완료: {address_matrix.shape}")
        print(f"✓ 지역명 유사도 행렬 생성: {self.address_similarity_matrix.shape}")

        # 5. 모든 특징 결합 (sparse matrix로 유지)
        self.feature_matrix = hstack([
            self.tfidf_matrix,
            self.category_matrix * 2,  # 카테고리 가중치 증가
            self.suitable_matrix * 2,  # 구성원 가중치 증가
        ])

        print(f"✓ 최종 특징 행렬: {self.feature_matrix.shape} (sparse)")

    def train_model(self):
        """
        모델 훈련 (유사도 행렬 계산)

        Content-Based Filtering 사용:
        - 코사인 유사도를 사용하여 장소 간 유사도 계산
        - 지역, 카테고리, 구성원, 카테고리 등을 종합적으로 고려
        """

        print("\n=== 모델 훈련 시작 ===")

        # 코사인 유사도 계산
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

        print(f"✓ 유사도 행렬 생성 완료: {self.similarity_matrix.shape}")
        print(f"✓ 평균 유사도: {self.similarity_matrix.mean():.4f}")
        print(f"✓ 최대 유사도: {self.similarity_matrix.max():.4f}")
        print(f"✓ 최소 유사도: {self.similarity_matrix.min():.4f}")

    def evaluate_model(self, k=10, max_km=15.0, min_cat_overlap=1, min_suit_overlap=1):
        """
        평가 기준(권장):
        - 거리: max_km km 이내
        - 카테고리 겹침: 최소 min_cat_overlap 개
        - 구성원 겹침: 최소 min_suit_overlap 개
        """
        print("\n=== 모델 평가 ===")
        print(f"📍 근접 기준: {max_km}km, 카테고리≥{min_cat_overlap}, 구성원≥{min_suit_overlap}, Top-{k}")

        # 벡터화 준비
        lat = np.radians(self.places_df['address_la'].values)
        lon = np.radians(self.places_df['address_lo'].values)
        cat_lists = self.places_df['category_list'].tolist()
        suit_lists = self.places_df['suitable_list'].tolist()

        precisions, recalls, ndcgs = [], [], []

        N = len(self.places_df)
        for idx in range(N):
            # --- 1) Ground truth 구축 ---
            # 거리 벡터 (idx 기준)
            dlat = lat - lat[idx]
            dlon = lon - lon[idx]
            a = np.sin(dlat/2)**2 + np.cos(lat[idx]) * np.cos(lat) * np.sin(dlon/2)**2
            dist_km = 2 * 6371 * np.arcsin(np.sqrt(a))

            # 카테고리/구성원 겹침
            cats_i = set(cat_lists[idx])
            suits_i = set(suit_lists[idx])

            cat_overlap = np.array([len(cats_i & set(cat_lists[j])) for j in range(N)])
            suit_overlap = np.array([len(suits_i & set(suit_lists[j])) for j in range(N)])

            mask_gt = (
                (dist_km <= max_km) &
                (cat_overlap >= min_cat_overlap) &
                (suit_overlap >= min_suit_overlap)
            )
            mask_gt[idx] = False  # 자기 자신 제외
            relevant_idx = np.where(mask_gt)[0].tolist()

            if not relevant_idx:
                continue  # 정답이 없는 샘플은 스킵

            # --- 2) 예측 Top-K ---
            sims = self.similarity_matrix[idx].copy()
            sims[idx] = -1
            topk = sims.argsort()[-k:][::-1]

            # --- 3) Metrics ---
            rel_in_topk = len(set(topk) & set(relevant_idx))
            precision = rel_in_topk / k
            recall = rel_in_topk / len(relevant_idx)

            # NDCG@k (이진 relevance)
            true_rel = np.zeros(N, dtype=int)
            true_rel[relevant_idx] = 1
            try:
                ndcg = ndcg_score([true_rel], [sims], k=k)
            except Exception:
                ndcg = 0.0

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        if not precisions:
            print("평가 가능한 샘플이 없습니다. 기준을 완화하세요.")
            return {'precision': 0, 'recall': 0, 'f1_score': 0, 'ndcg': 0}

        P, R, N = np.mean(precisions), np.mean(recalls), np.mean(ndcgs)
        F1 = 2*P*R/(P+R) if (P+R) > 0 else 0

        print(f"\n📊 평가 결과 (Top-{k}):")
        print(f"  - Precision@{k}: {P:.4f} (±{np.std(precisions):.4f})")
        print(f"  - Recall@{k}: {R:.4f} (±{np.std(recalls):.4f})")
        print(f"  - NDCG@{k}: {N:.4f} (±{np.std(ndcgs):.4f})")
        print(f"  - F1-Score: {F1:.4f}")
        print(f"\n📈 추가 분석:")
        print(f"  - 평가된 장소 수: {len(precisions)}")
        print(f"  - 우수(Precision≥0.8): {sum(p>=0.8 for p in precisions)}")
        print(f"  - 양호(0.6≤P<0.8): {sum((p>=0.6) & (p<0.8) for p in precisions)}")
        print(f"  - 보통(0.4≤P<0.6): {sum((p>=0.4) & (p<0.6) for p in precisions)}")
        print(f"  - 개선 필요(P<0.4): {sum(p<0.4 for p in precisions)}")

        return {'precision': P, 'recall': R, 'f1_score': F1, 'ndcg': N}


    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        두 지점 간의 거리를 계산 (km)
        Haversine 공식 사용
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # 지구 반경 (km)
        r = 6371

        return c * r

    def filter_by_location(self, center_lat, center_lon, radius_km):
        """
        위치 반경 내의 장소만 필터링

        Parameters:
        - center_lat: 중심 위도
        - center_lon: 중심 경도
        - radius_km: 반경 (km)

        Returns:
        - 필터링된 장소 인덱스 리스트
        """

        self.places_df['distances'] = self.places_df.apply(
            lambda row: self.haversine_distance(
                center_lat, center_lon,
                row['address_la'], row['address_lo']
            ),
            axis=1
        )

        within_radius = self.places_df['distances'] <= radius_km
        return self.places_df[within_radius].index.tolist()

    def recommend(self,
                  address=None,
                  suitables=None,
                  categorys=None,
                  center_location=None,
                  radius_km=51,
                  top_k=10):
        """
        여행지 추천

        Parameters:
        - suitables: 구성원 리스트 (예: ['가족', '아이'])
        - categorys: 카테고리 리스트 (예: ['자연', '사진'])
        - center_location: (위도, 경도) 튜플
        - radius_km: 반경 (km)
        - top_k: 추천 개수

        Returns:
        - 추천 장소 DataFrame
        """

        print("\n=== 추천 시작 ===")
        print(f"📍 위치: {address}")
        print(f"👥 구성원: {suitables}")
        print(f"🎨 카테고리: {categorys}")
        print(f"📍 중심 위치: ({center_location[0]:.4f}, {center_location[1]:.4f})")
        print(f"📏 반경: {radius_km}km")

        # 1. 필터링 시작
        candidate_indices = self.places_df.index.tolist()
        total_candidate_count = len(candidate_indices)
        distance_candidate_count = 0

        # recommend() 맨 앞에 추가
        if center_location is not None:
           lat, lon = center_location
           # lat은 |값|<=90, lon은 |값|<=180 이 정상
           if abs(lat) > 90 or abs(lon) > 180 or lat < 20 or lat > 50 or lon < 120 or lon > 140:
             # 한국 좌표 범위를 벗어나면 (lon,lat)로 들어왔다고 가정하고 스왑
             center_location = (lon, lat)

        # 위치 반경 필터링
        if center_location and radius_km:
            location_indices = self.filter_by_location(
                center_location[0], center_location[1], radius_km
            )
            candidate_indices = list(set(candidate_indices) & set(location_indices))
            distance_candidate_count = len(candidate_indices)
            print(f"  → 위치 반경 필터링 후: {len(candidate_indices)}개")

        if len(candidate_indices) == 0:
            print("⚠️  조건에 맞는 장소가 없습니다.")
            return pd.DataFrame()

        # 2. 스코어 계산
        scores = np.zeros(len(self.places_df))
        print(f"  → self.places_df : {self.places_df.columns}")

        # 구성원 매칭 점수 추가
        if suitables:
            for idx in candidate_indices:
                suitable = set(self.places_df.iloc[idx]['suitable_list'])
                suitable_match = len(set(suitables) & suitable) / len(suitables)
                scores[idx] += suitable_match * 0.5 # 가중치
        print(f"  → 구성원 매칭 점수 추가 후 점수 분포: {scores.mean():.4f} ± {scores.std():.4f}")

        # 카테고리 매칭 점수 추가
        if categorys:
            for idx in candidate_indices:
                place_categorys = set(self.places_df.iloc[idx]['category_list'])
                category_match = len(set(categorys) & place_categorys) / len(categorys)
                scores[idx] += category_match * 0.5 # 가중치
        print(f"  → 카테고리 매칭 점수 추가 후 점수 분포: {scores.mean():.4f} ± {scores.std():.4f}")

        # 거리 점수 추가
        if center_location and radius_km:
            distance_match = distance_candidate_count / total_candidate_count
            for idx in candidate_indices:
                
                if self.places_df.iloc[idx]['distances'] <= 3:
                    scores[idx] += distance_match * 0.6
                elif self.places_df.iloc[idx]['distances'] <= 10:
                    scores[idx] += distance_match * 0.5
                elif self.places_df.iloc[idx]['distances'] <= 20:
                    scores[idx] += distance_match * 0.4
                elif self.places_df.iloc[idx]['distances'] <= 30:
                    scores[idx] += distance_match * 0.3
                elif self.places_df.iloc[idx]['distances'] <= 40:
                    scores[idx] += distance_match * 0.2
                else:
                    scores[idx] += distance_match * 0.1
        print(f"  → 거리 매칭 점수 추가 후 점수 분포: {scores.mean():.4f} ± {scores.std():.4f}")

        # 3. 후보 중에서 Top-K 선택
        candidate_scores = [(idx, scores[idx]) for idx in candidate_indices]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, score in candidate_scores[:top_k]]

        # 4. 결과 생성
        result_df = self.places_df.iloc[top_indices].copy()
        result_df['recommendation_score'] = [scores[idx] for idx in top_indices]

        # 중심 위치가 있으면 거리 계산
        if center_location:
            result_df['distance_km'] = result_df.apply(
                lambda row: self.haversine_distance(
                    center_location[0], center_location[1],
                    row['address_la'], row['address_lo']
                ),
                axis=1
            ).round(2)

        print(f"\n✅ 추천 완료: {len(result_df)}개 장소")

        return result_df

    def display_recommendations(self, recommendations):
        """추천 결과 보기 좋게 출력"""

        if len(recommendations) == 0:
            print("\n추천 결과가 없습니다.")
            return

        print("\n" + "="*80)
        print("🎯 추천 여행지".center(80))
        print("="*80)

        for idx, (i, row) in enumerate(recommendations.iterrows(), 1):
            print(f"\n{idx}. {row['name']}")
            print(f"   📍 위치: {row['address']}")
            print(f"   🏷️  카테고리: {row['category']}")
            if 'categorys' in row:
                print(f"   🎨 카테고리: {row['categorys']}")
            print(f"   👥 적합: {row['suitable']}")
            if 'rating' in row:
                print(f"   ⭐ 평점: {row['rating']}")
            print(f"   📊 추천 점수: {row['recommendation_score']:.3f}")
            if 'distance_km' in row:
                print(f"   📏 거리: {row['distance_km']}km")

        print("\n" + "="*80)

    def save_model(self, model_path='recommendation_model.pkl'):
        """
        훈련된 모델 저장
        
        Parameters:
        - model_path: 저장할 파일 경로
        """
        try:
            # Sparse matrix를 각 컴포넌트로 저장 (메모리 최적화)
            model_data = {
                'similarity_matrix': self.similarity_matrix,
                # feature_matrix는 저장하지 않음 - 각 컴포넌트만 저장
                'tfidf_matrix': self.tfidf_matrix,
                'category_matrix': self.category_matrix,
                'suitable_matrix': self.suitable_matrix,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'mlb_category': self.mlb_category,
                'mlb_suitable': self.mlb_suitable,
                'scaler': self.scaler,
                'places_df': self.places_df,
                'address_similarity_matrix': self.address_similarity_matrix,
                'address_vectorizer': self.address_vectorizer
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # 파일 크기 확인
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            print(f"✓ 모델 저장 완료: {model_path}")
            print(f"  - 파일 크기: {file_size:.2f} MB")
            print(f"  - 유사도 행렬: {self.similarity_matrix.shape if self.similarity_matrix is not None else 'None'}")
            print(f"  - 지역명 유사도 행렬: {self.address_similarity_matrix.shape if self.address_similarity_matrix is not None else 'None'}")
            print(f"  - 장소 수: {len(self.places_df) if self.places_df is not None else 'None'}")
            return True
        except Exception as e:
            print(f"✗ 모델 저장 실패: {e}")
            return False

    def load_model(self, model_path='CBF_recommendation_model.pkl'):
        """
        저장된 모델 불러오기
        
        Parameters:
        - model_path: 불러올 파일 경로
        """
        try:
            if not os.path.exists(model_path):
                print(f"✗ 모델 파일이 존재하지 않습니다: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.similarity_matrix = model_data['similarity_matrix']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.mlb_category = model_data['mlb_category']
            self.mlb_suitable = model_data['mlb_suitable']
            self.scaler = model_data['scaler']
            self.places_df = model_data['places_df']
            
            # 각 컴포넌트 로드 (기존 feature_matrix 저장 방식과 호환성 유지)
            if 'tfidf_matrix' in model_data:
                # 새 버전: 각 컴포넌트를 로드하고 feature_matrix 재구성
                self.tfidf_matrix = model_data['tfidf_matrix']
                self.category_matrix = model_data['category_matrix']
                self.suitable_matrix = model_data['suitable_matrix']
                self.feature_matrix = hstack([
                    self.tfidf_matrix,
                    self.category_matrix * 2,
                    self.suitable_matrix * 2
                ])
            elif 'feature_matrix' in model_data:
                # 기존 버전: feature_matrix 직접 로드
                self.feature_matrix = model_data['feature_matrix']
                print("  ⚠️  기존 모델 형식입니다. 메모리 사용량이 클 수 있습니다.")
            else:
                raise ValueError("모델 파일에 feature 정보가 없습니다.")
            
            # 지역명 유사도 행렬 로드 (기존 모델과 호환성 유지)
            if 'address_similarity_matrix' in model_data:
                self.address_similarity_matrix = model_data['address_similarity_matrix']
                self.address_vectorizer = model_data.get('address_vectorizer', TfidfVectorizer())
            else:
                # 기존 모델의 경우 지역명 유사도 행렬 다시 계산
                print("  - 지역명 유사도 행렬 재계산 중...")
                self.address_vectorizer = TfidfVectorizer()
            
            # 파일 크기 확인
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            print(f"✓ 모델 로드 완료: {model_path}")
            print(f"  - 파일 크기: {file_size:.2f} MB")
            print(f"  - 유사도 행렬: {self.similarity_matrix.shape if self.similarity_matrix is not None else 'None'}")
            print(f"  - 지역명 유사도 행렬: {self.address_similarity_matrix.shape if self.address_similarity_matrix is not None else 'None'}")
            print(f"  - 장소 수: {len(self.places_df) if self.places_df is not None else 'None'}")
            return True
        except Exception as e:
            print(f"✗ 모델 로드 실패: {e}")
            return False