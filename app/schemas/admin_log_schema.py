from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict

class AdminLogBase(BaseModel):
    admin_id: int
    action: str              # "ban_user", "update_role", "delete_post" 등
    target_user: Optional[str] = None   # 영향을 받은 유저
    timestamp: datetime = datetime.utcnow()
    details: Optional[Dict] = None      # 세부 정보 (변경된 값, 사유 등)