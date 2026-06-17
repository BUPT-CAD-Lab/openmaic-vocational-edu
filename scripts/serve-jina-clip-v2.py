#!/usr/bin/env python3
"""OpenAI-compatible embedding server for local jina-clip-v2 testing.

This is a lightweight development helper for OpenMAIC's multimodal RAG flow.
It exposes:

  GET  /health
  POST /v1/embeddings

The request shape matches the local knowledge image embedding provider:

  {
    "model": "jina-clip-v2",
    "input": [
      {"type": "image", "image": "data:image/png;base64,...", "text": "caption"},
      "text query"
    ],
    "encoding_format": "float"
  }
"""

from __future__ import annotations

import argparse
import base64
import io
import traceback
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_PATH = "/mnt/hpfs/xiangc/llms/jina-clip-v2"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8300


class EmbeddingRequest(BaseModel):
    model: str = "jina-clip-v2"
    input: list[Any] | str
    encoding_format: str | None = "float"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve local jina-clip-v2 embeddings for OpenMAIC multimodal RAG.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. Defaults to cuda when available.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def load_image(value: str) -> Image.Image:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        data = base64.b64decode(value)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image input") from exc


def normalize_inputs(raw: Any) -> list[str | Image.Image]:
    items = raw if isinstance(raw, list) else [raw]
    normalized: list[str | Image.Image] = []
    for item in items:
        if isinstance(item, str):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            image = item.get("image") or item.get("image_url")
            if image:
                normalized.append(load_image(str(image)))
                continue
            text = item.get("text")
            if text:
                normalized.append(str(text))
                continue
        raise HTTPException(status_code=400, detail=f"Unsupported embedding input: {item!r}")
    return normalized


def disable_xformers_attention(model: SentenceTransformer) -> int:
    """Disable optional xFormers attention when xformers is not installed.

    jina-clip-v2's vision config can enable `xattn`. Without xformers installed,
    image embedding fails at runtime with:
      "Can't use xattn without xformers"

    For local testing, the standard attention path is slower but avoids requiring
    an xformers build that matches the active CUDA/PyTorch environment.
    """
    disabled = 0
    for module in model.modules():
        if hasattr(module, "xattn") and getattr(module, "xattn") is True:
            setattr(module, "xattn", False)
            disabled += 1
    return disabled


def create_app(model_path: str, device: str) -> FastAPI:
    app = FastAPI()
    model = SentenceTransformer(model_path, trust_remote_code=True)
    model = model.to(device)
    disabled_xattn_modules = disable_xformers_attention(model)

    @app.get("/health")
    def health() -> dict[str, str | int]:
        return {
            "status": "healthy",
            "model": "jina-clip-v2",
            "device": device,
            "disabled_xattn_modules": disabled_xattn_modules,
        }

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingRequest) -> dict[str, Any]:
        inputs = normalize_inputs(req.input)
        try:
            vectors = model.encode(
                inputs,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "object": "list",
            "model": req.model,
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector.astype(float).tolist(),
                }
                for index, vector in enumerate(vectors)
            ],
        }

    return app


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    app = create_app(args.model_path, device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
