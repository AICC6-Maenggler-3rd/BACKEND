from fastapi import APIRouter, Depends, HTTPException, Request, Query
import sys
from pathlib import Path
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.postgre_model import Place, Region
from app.repositories.placedb import get_place
from app.schemas.postgre_schema import PlaceSchema
from app.ml.contentBasedFiltering.main import TravelRecommendationSystem as CBF
from app.core.config import MODEL_DIR
from huggingface_hub import hf_hub_download

router = APIRouter()

# 전역 모델 인스턴스 (한 번만 로드)
_global_cbf_model = None

async def get_cbf_model():
    """CBF 모델을 싱글톤으로 관리"""
    global _global_cbf_model
    
    if _global_cbf_model is None:
        try:
            print("[INFO] CBF 모델 로딩 시작...")
            model_path = hf_hub_download(
                repo_id='JY1211/inpick-cbf',
                filename="CBF_recommendation_model.pkl",
                cache_dir=MODEL_DIR,
            )
            print(f"[INFO] 모델 다운로드 완료: {model_path}")
            
            model = CBF()
            loaded = model.load_model(model_path)
            if not loaded:
                raise Exception(f"Failed to load recommendation model from {model_path}")
            
            _global_cbf_model = model
            print("[INFO] CBF 모델 로딩 완료")
        except Exception as e:
            print(f"[ERROR] CBF 모델 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return _global_cbf_model

@router.get("/places")
async def recommend_places_from_user_input(
    address: str,
    suitables: list[str] = Query(default=[], alias="suitables[]"),
    categorys: list[str] = Query(default=[], alias="categorys[]"),
    center_location: list[float] = Query(alias="center_location[]"),
    db: AsyncSession = Depends(get_db)
):
    """사용자 입력에 기반한 content_based 장소 추천"""
    try:
        # 전역 모델 가져오기 (싱글톤)
        model = await get_cbf_model()
        
        # center_location은 쿼리에서 배열 형태로 들어오므로 튜플로 변환
        if center_location is None or len(center_location) != 2:
            raise HTTPException(status_code=422, detail="center_location must have exactly two numbers [lat, lon]")
        center_location_tuple = (float(center_location[0]), float(center_location[1]))

        recommendations = model.recommend(
                address=address,
                suitables=suitables,
                categorys=categorys,
                center_location=center_location_tuple,
        )
        print(f"[INFO] 추천 결과: {recommendations['place_id'].tolist()}")
        
        
        # 추천 결과에서 place_id 추출
        try:
            place_ids = recommendations['place_id'].tolist()
        except Exception:
            # fallback: dict 레코드 리스트로 가정
            place_ids = [rec['place_id'] for rec in recommendations]
        
        # 각 place_id에 대해 장소 상세 정보 가져오기
        places = []
        for place_id in place_ids:
            place = await get_place(db, place_id)
            if place:
                place_schema = PlaceSchema.model_validate(place)
                places.append(place_schema.model_dump())
        
        return places
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] recommend_places_from_user_input: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))