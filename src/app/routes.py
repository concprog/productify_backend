from fastapi import APIRouter

from fastapi import HTTPException
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from app import functions
from ai import


router = APIRouter()

DATA_STORE_PATH = "data/user_doc"


@router.post("/dailymap/")
async def make_notes(
    para: str,
):
    response = ai_functions.note_maker_summarize(para, n_paras=2)
    return ai_functions.generate_text_from_response(response)

@router.post("/roadmap")
async def search_similar_para(
    para: str,
):
    responses = ai_functions.search_passages(passage=para, top_k=4)
    response = {"paragraphs": responses}
    return response

