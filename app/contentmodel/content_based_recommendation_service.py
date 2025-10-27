import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.placedb import get_all_places
from app.repositories.regiondb import get_region_by_name, get_all_region
from app.schemas.postgre_schema import PlaceSchema
from typing import List, Dict, Any, Optional
import logging
import math
import re

logger = logging.getLogger(__name__)

class ContentBasedRecommendationService:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.place_features = None
        self.place_ids = None
        self.is_fitted = False
    
    async def fit(self, db: AsyncSession):
        """데이터베이스에서 장소 데이터를 가져와서 TF-IDF 벡터화 모델을 학습"""
        try:
            # 데이터베이스에서 모든 장소 정보 가져오기
            places = await get_all_places(db)
            
            if not places:
                logger.warning("No places found in database")
                return
            
            # 장소 데이터를 DataFrame 형태로 변환
            place_data = []
            for place in places:
                # 카테고리 정보를 문자열로 변환
                category_str = ''
                if hasattr(place, 'categories') and place.categories:
                    category_str = ' '.join([cat.name for cat in place.categories])
                
                place_data.append({
                    'place_id': place.place_id,
                    'place_name': place.name,
                    'place_type': place.type or '',
                    'category': category_str,
                    'description': place.description or '',
                    'address': place.address or '',
                    'address_la': place.address_la,
                    'address_lo': place.address_lo
                })
            
            df_places = pd.DataFrame(place_data)
            
            # 장소별 특성 텍스트 생성
            df_places["feature_text"] = (
                df_places["place_type"].fillna("") + " " + 
                df_places["category"].fillna("") + " " +
                df_places["description"].fillna("") + " " +
                df_places["address"].fillna("")
            )
            
            # TF-IDF 벡터화
            self.place_features = self.vectorizer.fit_transform(df_places["feature_text"])
            self.place_ids = df_places["place_id"].values
            
            # 장소 데이터 캐시 저장
            self.place_data_cache = place_data
            
            self.is_fitted = True
            logger.info(f"Content-based recommendation model fitted with {len(places)} places")
            
        except Exception as e:
            logger.error(f"Error fitting content-based recommendation model: {e}")
            raise
    
    def _create_user_profile(self, theme: str, relation: str, location: str = "") -> str:
        """사용자 프로필을 텍스트로 변환"""
        profile_parts = []
        
        if theme:
            profile_parts.append(theme)
        if relation:
            profile_parts.append(relation)
        if location:
            profile_parts.append(location)
        
        return " ".join(profile_parts)
    
    async def _filter_places_by_location(self, places_data: List[Dict], location: str, db: AsyncSession, radius_km: float = 50.0) -> List[Dict]:
        """지역명으로 장소 필터링 - 좌표 기반 방식으로 개선"""
        if not location:
            return places_data
        
        try:
            # 1. 데이터베이스에서 지역 좌표 가져오기
            region_data = await get_region_by_name(db, location)
            
            if not region_data:
                # 좌표를 찾을 수 없으면 키워드 검색으로 폴백
                logger.warning(f"Region '{location}' not found in database, falling back to keyword search")
                return self._filter_places_by_keyword(places_data, location)
            
            region_lat = float(region_data.get('address_la', 0))
            region_lng = float(region_data.get('address_lo', 0))
            
            if not region_lat or not region_lng:
                logger.warning(f"Invalid coordinates for region '{location}', falling back to keyword search")
                return self._filter_places_by_keyword(places_data, location)
            
            # 2. 지역 중심으로부터 반경 내의 장소 필터링
            filtered_places = []
            for place in places_data:
                place_lat = place.get('address_la')
                place_lng = place.get('address_lo')
                
                if not place_lat or not place_lng:
                    continue
                
                # Haversine 거리 계산
                distance = self._calculate_distance(region_lat, region_lng, place_lat, place_lng)
                
                if distance <= radius_km:
                    place['distance_from_center'] = distance
                    filtered_places.append(place)
            
            logger.info(f"Filtered {len(filtered_places)} places for location '{location}' (radius: {radius_km}km) from {len(places_data)} places")
            return filtered_places
            
        except Exception as e:
            logger.error(f"Error filtering places by location '{location}': {e}")
            return self._filter_places_by_keyword(places_data, location)
    
    def _filter_places_by_keyword(self, places_data: List[Dict], location: str) -> List[Dict]:
        """키워드 기반 필터링 (폴백용)"""
        filtered_places = []
        
        for place in places_data:
            address = place.get('address', '') or ''
            place_name = place.get('place_name', '') or ''
            
            # 텍스트 정규화
            place_text = f"{address} {place_name}".lower().replace('-', ' ')
            place_text = ' '.join(place_text.split())
            
            # 지역명이 포함되어 있는지 확인
            location_lower = location.lower()
            pattern = r'\b' + re.escape(location_lower) + r'\b'
            if re.search(pattern, place_text):
                filtered_places.append(place)
        
        logger.info(f"Filtered {len(filtered_places)} places for location '{location}' using keyword search")
        return filtered_places
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """두 좌표 간의 거리를 킬로미터로 계산 (Haversine 공식)"""
        R = 6371  # 지구의 반지름 (km)
        
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _cluster_places_by_distance(self, places_data: List[Dict], max_cluster_distance: float = 5.0) -> List[List[Dict]]:
        """거리 기반으로 장소들을 클러스터링"""
        if len(places_data) <= 1:
            return [places_data]
        
        # 좌표가 있는 장소들만 필터링
        valid_places = [p for p in places_data if 'address_la' in p and 'address_lo' in p]
        if len(valid_places) <= 1:
            return [places_data]
        
        # 좌표 배열 생성
        coordinates = np.array([[p['address_la'], p['address_lo']] for p in valid_places])
        
        # 클러스터 수 결정 (최대 5개, 최소 1개)
        n_clusters = min(5, max(1, len(valid_places) // 3))
        
        try:
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
            
            # 클러스터별로 장소 그룹화
            clusters = [[] for _ in range(n_clusters)]
            for i, place in enumerate(valid_places):
                cluster_id = cluster_labels[i]
                clusters[cluster_id].append(place)
            
            # 각 클러스터 내에서 거리 검증 및 필터링
            filtered_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    filtered_clusters.append(cluster)
                    continue
                
                # 클러스터 내 장소들 간의 최대 거리 계산
                max_distance = 0
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        dist = self._calculate_distance(
                            cluster[i]['address_la'], cluster[i]['address_lo'],
                            cluster[j]['address_la'], cluster[j]['address_lo']
                        )
                        max_distance = max(max_distance, dist)
                
                # 최대 거리가 허용 범위 내에 있으면 유지
                if max_distance <= max_cluster_distance:
                    filtered_clusters.append(cluster)
                else:
                    # 거리가 너무 멀면 개별 장소로 분리
                    filtered_clusters.extend([[place] for place in cluster])
            
            return filtered_clusters
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, returning original places")
            return [places_data]
    
    def _select_places_from_clusters(self, clusters: List[List[Dict]], num_recommendations: int) -> List[Dict]:
        """클러스터에서 장소들을 선택하여 추천 목록 생성 (음식점 1개 이상 보장, 거리 고려)"""
        selected_places = []
        has_restaurant = False
        
        # 클러스터를 크기 순으로 정렬 (큰 클러스터부터)
        sorted_clusters = sorted(clusters, key=len, reverse=True)
        
        for cluster in sorted_clusters:
            if len(selected_places) >= num_recommendations:
                break
            
            # 클러스터 내에서 음식점과 기타 장소 분류
            cluster_restaurants = []
            cluster_others = []
            
            # 음식점 키워드 정의
            restaurant_keywords = ['음식점', 'restaurant', '식당']
            
            for place in cluster:
                place_type = place.get('place_type', '').lower()
                category = place.get('category', '').lower()
                
                # place_type 또는 category에서 키워드 확인
                combined_text = f"{place_type} {category}"
                is_restaurant = any(keyword in combined_text for keyword in restaurant_keywords)
                
                # 음식점만 별도 카테고리로 분류 (카페는 일반 장소로 분류)
                if is_restaurant:
                    cluster_restaurants.append(place)
                else:
                    cluster_others.append(place)
            
            # 클러스터 내에서 유사도 순으로 정렬
            cluster_restaurants.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            cluster_others.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # 아직 음식점을 선택하지 않았고, 이 클러스터에 음식점이 있으면 우선 선택
            if not has_restaurant and cluster_restaurants and len(selected_places) < num_recommendations:
                selected_places.append(cluster_restaurants[0])
                has_restaurant = True
                cluster_restaurants = cluster_restaurants[1:]  # 선택된 음식점 제거
            
            # 클러스터 내에서 나머지 장소들 선택 (유사도 순으로 정렬)
            cluster_places = cluster_restaurants + cluster_others
            cluster_places.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            for place in cluster_places:
                if len(selected_places) >= num_recommendations:
                    break
                selected_places.append(place)
        
        logger.info(f"Selected {len(selected_places)} places from {len(clusters)} clusters (requested: {num_recommendations})")
        return selected_places[:num_recommendations]
    
    async def recommend_places(
        self, 
        db: AsyncSession,
        theme: str, 
        relation: str, 
        location: str = "",
        num_recommendations: int = 5,
        exclude_place_ids: List[int] = None
    ) -> List[Dict[str, Any]]:
        """사용자 특성에 기반하여 장소를 추천"""
        
        if not self.is_fitted:
            await self.fit(db)
        
        if not self.is_fitted:
            logger.error("Model not fitted, cannot make recommendations")
            return []
        
        try:
            # 사용자 프로필 생성
            user_profile = self._create_user_profile(theme, relation, location)
            
            # 사용자 프로필을 TF-IDF 벡터로 변환
            user_vector = self.vectorizer.transform([user_profile])
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(user_vector, self.place_features).ravel()
            
            # 장소 데이터와 유사도 점수를 결합
            place_similarities = []
            for i, place_id in enumerate(self.place_ids):
                # 원본 장소 데이터에서 해당 장소 찾기
                place_data = None
                for place in self.place_data_cache:
                    if place['place_id'] == place_id:
                        place_data = place.copy()
                        break
                
                if place_data:
                    place_data['similarity_score'] = similarities[i]
                    place_similarities.append(place_data)
            
            # 지역 기반 필터링 적용
            if location:
                place_similarities_before = len(place_similarities)
                place_similarities = await self._filter_places_by_location(place_similarities, location, db)
                logger.info(f"Location filtering: {place_similarities_before} -> {len(place_similarities)} places for location: {location}")
            
            # 제외할 장소 필터링
            if exclude_place_ids:
                exclude_count_before = len(place_similarities)
                place_similarities = [
                    place for place in place_similarities 
                    if place['place_id'] not in exclude_place_ids
                ]
                logger.info(f"Exclude filtering: {exclude_count_before} -> {len(place_similarities)} places (excluded {len(exclude_place_ids)} places)")
            
            # 유사도 점수로 정렬
            place_similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # 상위 30개 장소만 클러스터링에 사용 (성능 및 품질 개선)
            top_places = place_similarities[:min(30, len(place_similarities))]
            logger.info(f"Using top {len(top_places)} places for clustering (from {len(place_similarities)} total)")
            
            # 거리 기반 클러스터링 적용
            clusters = self._cluster_places_by_distance(top_places, max_cluster_distance=5.0)
            logger.info(f"Created {len(clusters)} clusters from {len(top_places)} places")
            
            # 클러스터에서 장소 선택
            selected_places = self._select_places_from_clusters(clusters, num_recommendations)
            
            # 클러스터링 후 선택된 장소가 부족한 경우, 상위 유사도 장소를 추가
            if len(selected_places) < num_recommendations and len(place_similarities) > 0:
                remaining_needed = num_recommendations - len(selected_places)
                selected_place_ids = {p['place_id'] for p in selected_places}
                
                # 이미 선택된 장소를 제외하고 상위 유사도 장소 추가
                for place in place_similarities:
                    if place['place_id'] not in selected_place_ids:
                        selected_places.append(place)
                        selected_place_ids.add(place['place_id'])
                        remaining_needed -= 1
                        if remaining_needed <= 0:
                            break
                
                logger.info(f"Added {num_recommendations - len(selected_places) + remaining_needed} more places from top similarity scores")
            
            # 추천 결과 생성
            recommendations = []
            for place in selected_places:
                if place['similarity_score'] > 0:  # 점수가 0보다 큰 경우만
                    recommendations.append({
                        'place_id': int(place['place_id']),
                        'similarity_score': float(place['similarity_score']),
                        'theme': theme,
                        'relation': relation,
                        'location': location
                    })
            
            logger.info(f"Generated {len(recommendations)} recommendations for theme: {theme}, relation: {relation}, location: {location}")
            
            # 디버그: 추천이 0개인 경우 상세 로그
            if len(recommendations) == 0:
                logger.warning(f"No recommendations generated. Selected places: {len(selected_places)}, "
                             f"Filtered places: {len(place_similarities)}, "
                             f"Excluded: {len(exclude_place_ids) if exclude_place_ids else 0} places")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_type': 'content_based_recommendation',
            'is_fitted': self.is_fitted,
            'num_places': len(self.place_ids) if self.place_ids is not None else 0,
            'vectorizer_features': self.vectorizer.get_feature_names_out().tolist() if self.is_fitted else []
        }

# 전역 인스턴스
content_based_service = ContentBasedRecommendationService()
