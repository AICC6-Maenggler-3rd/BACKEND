from fastapi import APIRouter, Depends, HTTPException
from app.repositories import userdb, placedb, categorydb
from app.db.postgresql import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.place_schema import PlaceListResponse
from app.schemas.category_schema import CategorySchema, CategoryListResponse, CategoryBase
from fastapi import HTTPException
from sqlalchemy import select, update, delete
from app.models.postgre_model import Category, Place
from datetime import datetime, timezone
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
        
        # 전체 장소 수 조회
        all_places = await placedb.get_places(db)
        total_places_count = len(all_places)
        
        # 각 계정의 상세 정보 조회
        accounts_with_details = []
        sns_places_count = 0
        for nickname in nicknames:
            places = await placedb.get_places_by_insta_nickname(db, nickname, 1, 100000)
            
            # 상태 확인 (첫 번째 장소의 deleted_at 기준)
            is_active = True
            last_activity = None
            if places.places:
                first_place = places.places[0]
                is_active = first_place.deleted_at is None
                last_activity = first_place.updated_at.isoformat() if first_place.updated_at else None
            
            account_places_count = len(places.places)
            sns_places_count += account_places_count
            
            accounts_with_details.append({
                "nickname": nickname,
                "places_count": account_places_count,
                "status": "active" if is_active else "inactive",
                "last_activity": last_activity
            })
        
        return {
            "accounts": accounts_with_details,
            "total_count": len(accounts_with_details),
            "total_places_count": total_places_count,
            "sns_places_count": sns_places_count
        }
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
            places = await placedb.get_places_by_insta_nickname(db, nickname, 1, 100000)  # 모든 장소 조회
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

@router.patch("/sns-accounts/{old_nickname}")
async def update_sns_account(old_nickname: str, account_data: dict, db: AsyncSession = Depends(get_db)):
    """SNS 계정 닉네임 수정 (병합 지원)"""
    try:
        new_nickname = account_data.get("nickname")
        force_merge = account_data.get("force_merge", False)  # 병합 강제 옵션
        if not new_nickname:
            raise HTTPException(status_code=400, detail="새 닉네임은 필수입니다.")
        
        print(f"[DEBUG] SNS 계정 수정 요청: '{old_nickname}' -> '{new_nickname}'")
        
        # 기존 계정 존재 확인 (모든 장소 조회 - 페이지네이션 제한 없음)
        existing_places = await placedb.get_places_by_insta_nickname(db, old_nickname, 1, 100000)
        if not existing_places.places:
            raise HTTPException(status_code=404, detail="해당 SNS 계정을 찾을 수 없습니다.")
        
        print(f"[DEBUG] 기존 계정의 장소 수: {len(existing_places.places)}")
        print(f"[WARNING] 1000개 이상의 장소가 있을 경우 이전에 일부만 업데이트되었을 수 있습니다.")
        
        # 새 닉네임 중복 확인 (old_nickname과 다를 때만)
        if old_nickname != new_nickname:
            # 현재 수정하려는 장소들의 ID 목록
            current_place_ids = [place.place_id for place in existing_places.places]
            print(f"[DEBUG] 현재 계정의 장소 ID 목록: {current_place_ids}")
            
            # 새 닉네임으로 조회 (모든 장소)
            duplicate_places = await placedb.get_places_by_insta_nickname(db, new_nickname, 1, 100000)
            
            print(f"[DEBUG] 새 닉네임으로 조회된 장소 수: {len(duplicate_places.places)}")
            if duplicate_places.places:
                duplicate_place_ids = [p.place_id for p in duplicate_places.places]
                print(f"[DEBUG] 새 닉네임의 장소 ID 목록: {duplicate_place_ids}")
            
            # 현재 수정하려는 장소들을 제외하고 중복 확인
            other_places = [p for p in duplicate_places.places if p.place_id not in current_place_ids]
            
            print(f"[DEBUG] 다른 장소 수 (현재 계정 제외): {len(other_places)}")
            if other_places and not force_merge:
                other_place_ids = [p.place_id for p in other_places]
                other_place_names = [p.name for p in other_places[:3]]  # 처음 3개만
                print(f"[DEBUG] 다른 장소 ID 목록: {other_place_ids}")
                print(f"[DEBUG] 다른 장소 이름 (샘플): {other_place_names}")
                
                detail_msg = f"'{new_nickname}' 닉네임은 이미 다른 {len(other_places)}개의 장소에서 사용 중입니다. "
                detail_msg += f"(예: {', '.join(other_place_names[:2])}) "
                detail_msg += "계정을 병합하려면 병합 옵션을 사용하세요."
                raise HTTPException(status_code=400, detail=detail_msg)
            elif other_places and force_merge:
                print(f"[INFO] 병합 모드: {len(other_places)}개의 기존 장소와 {len(existing_places.places)}개의 새 장소를 병합합니다.")
        
        # 모든 관련 장소의 닉네임 업데이트
        for place in existing_places.places:
            await db.execute(
                update(Place)
                .where(Place.place_id == place.place_id)
                .values(
                    insta_nickname=new_nickname,
                    updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
                )
            )
        
        await db.commit()
        
        print(f"[DEBUG] SNS 계정 수정 완료: {len(existing_places.places)}개 장소 업데이트")
        
        return {
            "message": f"SNS 계정이 '{old_nickname}'에서 '{new_nickname}'으로 성공적으로 수정되었습니다.",
            "old_nickname": old_nickname,
            "new_nickname": new_nickname,
            "affected_places": len(existing_places.places)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"SNS 계정 수정 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sns-accounts/{insta_nickname}")
async def delete_sns_account(insta_nickname: str, db: AsyncSession = Depends(get_db)):
    """SNS 계정 삭제"""
    try:
        # 해당 닉네임의 장소들 조회 (모든 장소)
        places = await placedb.get_places_by_insta_nickname(db, insta_nickname, 1, 100000)
        
        if not places.places:
            raise HTTPException(status_code=404, detail="해당 SNS 계정을 찾을 수 없습니다.")
        
        # 모든 관련 장소 삭제 (soft delete)
        for place in places.places:
            await db.execute(
                update(Place)
                .where(Place.place_id == place.place_id)
                .values(deleted_at=datetime.now(timezone.utc).replace(tzinfo=None))
            )
        
        await db.commit()
        
        return {
            "message": f"SNS 계정 '{insta_nickname}'이 성공적으로 삭제되었습니다.",
            "nickname": insta_nickname,
            "affected_places": len(places.places)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"SNS 계정 삭제 API 에러: {e}")
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

@router.patch("/users/{user_id}/status")
async def toggle_user_status(user_id: int, db: AsyncSession = Depends(get_db)):
    """사용자 계정 상태 변경 (활성화/비활성화)"""
    try:
        # 사용자 존재 여부 확인
        user = await userdb.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        
        # 현재 상태에 따라 반대 상태로 변경
        new_status = "active" if user.status == "deactive" else "deactive"
        
        # 상태 업데이트
        success = await userdb.update_user_status(db, user_id, new_status)
        if not success:
            raise HTTPException(status_code=500, detail="사용자 상태 변경에 실패했습니다.")
        
        status_text = "활성화" if new_status == "active" else "비활성화"
        return {"message": f"사용자가 성공적으로 {status_text}되었습니다.", "new_status": new_status}
    except HTTPException:
        raise
    except Exception as e:
        print(f"사용자 상태 변경 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/activity-logs")
async def get_user_activity_logs(user_id: int, db: AsyncSession = Depends(get_db)):
    """사용자 활동 로그 조회"""
    try:
        # 사용자 존재 여부 확인
        user = await userdb.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        
        # 활동 로그 조회 (임시로 가짜 데이터 반환)
        activity_logs = [
            {
                "id": 1,
                "action": "로그인",
                "description": "시스템에 로그인했습니다.",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "created_at": "2024-01-15T10:30:00Z"
            },
            {
                "id": 2,
                "action": "프로필 수정",
                "description": "프로필 정보를 수정했습니다.",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "created_at": "2024-01-15T09:15:00Z"
            },
            {
                "id": 3,
                "action": "장소 검색",
                "description": "서울 강남구에서 카페를 검색했습니다.",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "created_at": "2024-01-14T16:45:00Z"
            },
            {
                "id": 4,
                "action": "로그아웃",
                "description": "시스템에서 로그아웃했습니다.",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "created_at": "2024-01-14T18:20:00Z"
            },
            {
                "id": 5,
                "action": "로그인",
                "description": "시스템에 로그인했습니다.",
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "created_at": "2024-01-14T08:30:00Z"
            }
        ]
        
        return {
            "user_id": user_id,
            "user_name": user.name,
            "user_email": user.email,
            "activity_logs": activity_logs,
            "total_count": len(activity_logs)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"사용자 활동 로그 조회 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/users/{user_id}")
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    """사용자 계정 삭제"""
    try:
        # 사용자 존재 여부 확인
        user = await userdb.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        
        # 사용자 삭제
        success = await userdb.delete_user(db, user_id)
        if not success:
            raise HTTPException(status_code=500, detail="사용자 삭제에 실패했습니다.")
        
        return {"message": "사용자가 성공적으로 삭제되었습니다."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"사용자 삭제 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 카테고리 관리 API ====================

@router.get("/categories")
async def get_categories_list(db: AsyncSession = Depends(get_db)):
    """카테고리 목록 조회"""
    try:
        categories = await categorydb.get_categories(db)
        
        # 각 카테고리별 장소 수 계산
        category_list = []
        for category in categories:
            # 해당 카테고리의 장소 수 조회
            places = await placedb.get_places_by_category(db, category.category_id)
            place_count = len(places) if places else 0
            
            category_data = {
                "category_id": category.category_id,
                "name": category.name,
                "status": category.status,
                "created_at": category.created_at.isoformat() if category.created_at else None,
                "updated_at": category.updated_at.isoformat() if category.updated_at else None,
                "place_count": place_count
            }
            category_list.append(category_data)
        
        return {
            "categories": category_list,
            "total_count": len(category_list)
        }
    except Exception as e:
        print(f"카테고리 목록 조회 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/categories")
async def create_category(category_data: CategoryBase, db: AsyncSession = Depends(get_db)):
    """새 카테고리 생성"""
    try:
        # 중복 카테고리명 확인
        existing_category = await db.execute(
            select(Category).where(Category.name == category_data.name)
        )
        if existing_category.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="이미 존재하는 카테고리명입니다.")
        
        # 새 카테고리 생성
        current_time = datetime.now(timezone.utc).replace(tzinfo=None)
        new_category = Category(
            name=category_data.name,
            status="active",  # 항상 활성으로 생성
            created_at=current_time,
            updated_at=current_time
        )
        
        db.add(new_category)
        await db.commit()
        await db.refresh(new_category)
        
        return {
            "message": "카테고리가 성공적으로 생성되었습니다.",
            "category": {
                "category_id": new_category.category_id,
                "name": new_category.name,
                "status": new_category.status,
                "created_at": new_category.created_at.isoformat(),
                "place_count": 0
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"카테고리 생성 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/categories/{category_id}")
async def update_category(category_id: int, category_data: CategoryBase, db: AsyncSession = Depends(get_db)):
    """카테고리 수정"""
    try:
        # 카테고리 존재 여부 확인
        category = await db.execute(
            select(Category).where(Category.category_id == category_id)
        )
        category = category.scalar_one_or_none()
        if not category:
            raise HTTPException(status_code=404, detail="카테고리를 찾을 수 없습니다.")
        
        # 중복 카테고리명 확인 (자기 자신 제외)
        existing_category = await db.execute(
            select(Category).where(
                Category.name == category_data.name,
                Category.category_id != category_id
            )
        )
        if existing_category.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="이미 존재하는 카테고리명입니다.")
        
        # 카테고리 업데이트
        current_time = datetime.now(timezone.utc).replace(tzinfo=None)
        await db.execute(
            update(Category)
            .where(Category.category_id == category_id)
            .values(
                name=category_data.name,
                status="active",  # 항상 활성으로 유지
                updated_at=current_time
            )
        )
        await db.commit()
        
        return {"message": "카테고리가 성공적으로 수정되었습니다."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"카테고리 수정 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/categories/{category_id}")
async def delete_category(category_id: int, db: AsyncSession = Depends(get_db)):
    """카테고리 삭제"""
    try:
        # 카테고리 존재 여부 확인
        category = await db.execute(
            select(Category).where(Category.category_id == category_id)
        )
        category = category.scalar_one_or_none()
        if not category:
            raise HTTPException(status_code=404, detail="카테고리를 찾을 수 없습니다.")
        
        # 해당 카테고리에 속한 장소가 있는지 확인
        places = await placedb.get_places_by_category(db, category_id)
        if places and len(places) > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"이 카테고리에는 {len(places)}개의 장소가 있어서 삭제할 수 없습니다. 먼저 장소를 다른 카테고리로 이동하거나 삭제해주세요."
            )
        
        # 카테고리 삭제
        await db.execute(
            delete(Category).where(Category.category_id == category_id)
        )
        await db.commit()
        
        return {"message": "카테고리가 성공적으로 삭제되었습니다."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"카테고리 삭제 API 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))