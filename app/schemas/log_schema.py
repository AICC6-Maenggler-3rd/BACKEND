from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict

class UserLogSchema(BaseModel):
    user_id: int
    action: str
    ip: str
    user_agent: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    extra: Optional[Dict] = None