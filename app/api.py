from fastapi import FastAPI
from translation_impl import translate
from network_proto import TranslationRequest, TranslationResponse

app = FastAPI()


@app.post("/translation")
async def translation(req: TranslationRequest) -> TranslationResponse:
    return translate(req)
