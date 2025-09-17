from fastapi import APIRouter, Depends, HTTPException, Request

router = APIRouter()

@router.get("/{example}")
async def find_document_list(example : str, request: Request):
    try:
      return {"message":f"example {example}"}
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"처리 실패: {str(e)}")