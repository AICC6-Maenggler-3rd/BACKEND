from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict

class ActivityLogBase(BaseModel):
    user_id: str
    action: str          # "login", "logout" 등
    timestamp: datetime = datetime.utcnow()
    metadata: Optional[Dict] = None   # 추가정보 (게시글 ID, IP 등)