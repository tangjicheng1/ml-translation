from pydantic import BaseModel
from typing import List


class SingleRecord(BaseModel):
    id: str
    text: str


class RequestBody(BaseModel):
    fromLang: str
    toLang: str
    records: List[SingleRecord]


class TranslationRequest(BaseModel):
    payload: RequestBody


class TranslationResponse(BaseModel):
    result: List[SingleRecord]
