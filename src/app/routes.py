from fastapi import APIRouter

from fastapi import HTTPException
from fastapi import UploadFile, File
from fastapi.responses import FileResponse

from app import functions
from ai import chains


router = APIRouter()

DATA_STORE_PATH = "data/user_doc"


@router.post("/dailymap/")
async def get_daily_map(
    goal: str,
):
    response = chains.run_task_decomp(goal)
    return response


@router.post("/roadmap/")
async def search_similar_para(goal: str, background: str, expectations: str):
    response = chains.run_roadgen(goal, background)
    response = {"nodes": response}
    return response
