from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Literal

# 탈퇴 사유 수집
class UserDeleteRequest(BaseModel):
  reason: Optional[str] = None

# 응답
class UserDeleteResponse(BaseModel):
  success: bool
  message: str
  status: str
  deactivated_at: Optional[datetime] = None