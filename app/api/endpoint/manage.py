from fastapi import APIRouter, Depends, HTTPException
from app.repositories import userdb, placedb, categorydb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.place_schema import PlaceListResponse
router = APIRouter()

@router.get("/dashboard")
async def get_dashboard_data(db: AsyncSession = Depends(get_db)):
    users = await userdb.get_users(db)
    users_count = len(users)
    places = await placedb.get_places(db)
    places_count = len(places)
    categories = await categorydb.get_categories(db)
    categories_count = len(categories)
    # SNS 계정 수 조회
    sns_nicknames = await placedb.get_insta_nicknames(db)
    sns_accounts_count = len(sns_nicknames)
    
    print(f'users_count, places_count, categories_count, sns_accounts_count : {users_count}, {places_count}, {categories_count}, {sns_accounts_count}')

    return {
        "users_count": users_count, 
        "places_count": places_count, 
        "categories_count": categories_count,
        "sns_accounts_count": sns_accounts_count,
        "users": users  # 사용자 목록도 함께 반환
    }

@router.get("/users")
async def get_users_list(db: AsyncSession = Depends(get_db)):
    """사용자 목록 조회"""
    try:
        users = await userdb.get_users(db)
        return users
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sns-accounts")
async def get_sns_accounts(db: AsyncSession = Depends(get_db)):
    """SNS 계정 관리 - 인스타그램 닉네임 목록 조회"""
    try:
        nicknames = await placedb.get_insta_nicknames(db)
        return {"insta_nicknames": nicknames, "total_count": len(nicknames)}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sns-accounts/{insta_nickname}/places")
async def get_places_by_sns_account(insta_nickname: str, page: int = 1, limit: int = 30, db: AsyncSession = Depends(get_db)) -> PlaceListResponse:
    """특정 SNS 계정의 장소 목록 조회"""
    try:
        place_list = await placedb.get_places_by_insta_nickname(db, insta_nickname, page, limit)
        return place_list
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sns-accounts/stats")
async def get_sns_accounts_stats(db: AsyncSession = Depends(get_db)):
    """SNS 계정 통계 정보 조회"""
    try:
        nicknames = await placedb.get_insta_nicknames(db)
        total_accounts = len(nicknames)
        
        # 각 계정별 장소 수 계산
        account_stats = []
        for nickname in nicknames:
            places = await placedb.get_places_by_insta_nickname(db, nickname, 1, 1000)  # 큰 limit으로 모든 장소 조회
            account_stats.append({
                "nickname": nickname,
                "places_count": len(places.places)
            })
        
        return {
            "total_accounts": total_accounts,
            "account_stats": account_stats
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/places/popularity")
async def get_places_popularity(db: AsyncSession = Depends(get_db)):
    """장소 인기도 분석 - 방문 횟수 기준 TOP 장소"""
    try:
        print("장소 인기도 API 호출됨")
        places = await placedb.get_places(db)
        print(f"조회된 장소 수: {len(places)}")
        
        # count 기준으로 정렬하여 인기 장소 추출
        popular_places = sorted(places, key=lambda x: x.count, reverse=True)[:10]
        print(f"인기 장소 수: {len(popular_places)}")
        
        # 차트용 데이터 포맷
        chart_data = []
        for place in popular_places:
            chart_data.append({
                "place_id": place.place_id,
                "name": place.name,
                "count": place.count,
                "address": place.address,
                "type": place.type,
                "insta_nickname": place.insta_nickname
            })
        
        result = {
            "popular_places": chart_data,
            "total_places": len(places)
        }
        print(f"반환할 데이터: {result}")
        return result
    except Exception as e:
        print(f"장소 인기도 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/places/category-stats")
async def get_places_category_stats(db: AsyncSession = Depends(get_db)):
    """카테고리별 장소 통계"""
    try:
        print("카테고리 통계 API 호출됨")
        places = await placedb.get_places(db)
        categories = await categorydb.get_categories(db)
        print(f"조회된 장소 수: {len(places)}")
        print(f"조회된 카테고리 수: {len(categories)}")
        
        # 카테고리별 장소 수 계산
        category_stats = {}
        for place in places:
            # place.categories는 relationship이므로 실제로는 조인 쿼리가 필요할 수 있음
            # 일단 place.type을 사용
            place_type = place.type
            if place_type not in category_stats:
                category_stats[place_type] = 0
            category_stats[place_type] += 1
        
        print(f"카테고리별 통계: {category_stats}")
        
        # 차트용 데이터 포맷
        chart_data = []
        for category_name, count in category_stats.items():
            chart_data.append({
                "category": category_name,
                "count": count
            })
        
        # count 기준으로 정렬
        chart_data.sort(key=lambda x: x["count"], reverse=True)
        
        result = {
            "category_stats": chart_data,
            "total_categories": len(category_stats)
        }
        print(f"반환할 데이터: {result}")
        return result
    except Exception as e:
        print(f"카테고리 통계 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))