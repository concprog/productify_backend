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
    response = chains.llm_to_json(response["output"])
    return response


@router.post("/roadmap/")
async def get_roadmap(goal: str, background: str, expectations: str):
    response = chains.run_roadgen(goal, background, expectations)
    response = chains.llm_to_json(response["output"])
    return response
