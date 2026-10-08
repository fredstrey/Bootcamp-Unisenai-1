from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from processing import process_image_bytes


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Contador de Parafusos", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.post("/api/process")
async def api_process(file: UploadFile = File(...)) -> Response:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie uma imagem valida.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        result = await asyncio.to_thread(process_image_bytes, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ok, buffer = cv2.imencode(".png", result.composite)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao gerar a imagem de saida.")

    headers = {
        "X-Screw-Count": str(result.count),
        "X-Rejected-Count": str(result.rejected),
    }
    return Response(content=buffer.tobytes(), media_type="image/png", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)