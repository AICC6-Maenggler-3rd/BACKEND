from fastapi import APIRouter, Depends, HTTPException, Request
import sys
from pathlib import Path
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.postgre_model import Place, Region
from app.api.function.common import haversine_distance

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent 
root_dir = current_dir.parent.parent.parent
place_file = root_dir / 'uploads' / 'place.csv'
dummy_file = root_dir / 'uploads' / 'dummy_data'/ 'cf_ratings.csv'
region_file = root_dir / 'uploads' / 'region.csv'
sys.path.append(str(root_dir))

router = APIRouter()

@router.get("/cf/place")
async def make_cf_place_recommendation(db: AsyncSession = Depends(get_db)):
    """
    Collaborative Filtering 기반 장소 추천
    
    Args:
        더미데이터 : uploads/dummy_data/cf_ratings.csv
        user_id,place_id,relation,location,theme,duration,rating
        1127,4546,친구와,전라남도 고흥군 과역면 호덕리,바다,1박2일,3.3

        지역데이터 : uploads/region.csv
        name,address_lo,address_la
        서울특별시,126.978652258309,37.5668260046608

        장소데이터 : select * from place 
    Returns:
        dict: 장소 추천 결과
    """

    try:
        # DB에서 장소 데이터 가져오기
        query = select(Place.place_id, Place.address_la, Place.address_lo)
        result = await db.execute(query)
        place_data = result.fetchall()

        df_ratings = pd.read_csv(dummy_file)
        df_region = pd.read_csv(region_file)
        # DataFrame으로 변환
        df_place = pd.DataFrame(place_data, columns=['place_id', 'address_la', 'address_lo'])

        # merge 수행
        df_result = df_ratings.merge(
        df_region,
        left_on='location',
        right_on='name',
        how='left'
        ).drop(columns=['name'])
        df_result = df_result.rename(columns={
            'address_lo': 'location_lo',
            'address_la': 'location_la'
        })

        # place merge
        df_result = df_result.merge(
            df_place,
            on='place_id',
            how='left'
        ).rename(columns={
            'address_lo': 'place_lo',
            'address_la': 'place_la'
        })

        # 거리 계산
        df_result['distance'] = df_result.apply(
            lambda row: haversine_distance(
                row['location_la'], row['location_lo'], row['place_la'], row['place_lo']
                ), axis=1)

        # # 거리 제한을 더욱 강화
        # if distance > 200:
        #     distance_score = -20  # 200km 이상은 매우 큰 마이너스 점수
        # elif distance > 150:
        #     distance_score = -10  # 150-200km는 큰 마이너스 점수
        # elif distance > 100:
        #     distance_score = 0    # 100-150km는 중립
        # elif distance > 50:
        #     distance_score = 3    # 50-100km는 보통 점수
        # else:
        #     distance_score = 8    # 50km 이내는 매우 높은 점수
        print("========================================")
        print("df_result", df_result['distance'])
        print("df_result", df_result['user_id'].max())
        print("df_result", df_result['place_id'].max())
        print("df_result", df_result['user_id'].min())
        print("df_result", df_result['place_id'].min())
        print("df_result", df_result.shape)
        print("df_result", df_result.columns)
        print("df_result", df_result.loc[:, ['place_id', 'location','distance']])
        print("========================================")
        
    except Exception as e:
        return {'error' : str(e)}
    return 'success'

@router.get('/test/distance')
async def test_distance(db: AsyncSession = Depends(get_db)):
    try:
        df_place = pd.read_csv(place_file)
        df_place['suitable']= df_place['category'].apply(lambda x : x.split(','))
    except Exception as e:
        return {'error' : str(e)}
    return 'success'