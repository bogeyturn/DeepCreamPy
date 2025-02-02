import os
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

class DecensorItem(BaseModel):
    img_id: str
    """Images were uploaded earlier."""
    variations: int
    """How many variations to generate"""
    is_moasic: bool
    """Which model to use"""
    mask: str
    """RGB mask or file"""
    output: Optional[str]
    """None will return to web"""


class DecensorRequest(BaseModel):
    imgs: list[DecensorItem]

