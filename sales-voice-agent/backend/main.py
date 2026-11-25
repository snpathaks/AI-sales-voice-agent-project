"""
backend/main.py
===============
Sales Voice Agent — FastAPI backend

Uses the modern `google-genai` SDK (v1.x / google-genai package)
which is compatible with protobuf >=7 and Python 3.14.

Pipeline
--------
Text / Audio → RAG (numpy cosine) → Gemini Flash → reply text + (optional TTS)
"""

from __future__ import annotations

import base64
import os
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types as genai_types

# ── Load env ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Constants ───────────────────────────────────────────────────────────────
KB_PATH       = Path(__file__).parent.parent / "kb" / "company.md"
EMBED_MODEL   = "gemini-embedding-001"
CHAT_MODEL    = "gemini-2.0-flash"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 4

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Sales Voice Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════════════════
#  RAG — pure-numpy cosine vector store (zero extra deps)
# ════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def _embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Embed a list of texts via google-genai → (N, D) float32, L2-normalised."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    vecs = np.array([e.values for e in result.embeddings], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


class NumpyVectorStore:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        # embed in batches of 20
        parts: list[np.ndarray] = []
        for i in range(0, len(chunks), 20):
            parts.append(_embed(chunks[i : i + 20]))
        self._matrix = np.vstack(parts)
        print(f"[RAG] Indexed {len(chunks)} chunks from knowledge base")

    def retrieve(self, query: str, k: int = TOP_K) -> list[str]:
        q = _embed([query], task_type="RETRIEVAL_QUERY")[0]
        scores  = self._matrix @ q
        top_k   = int(min(k, len(self.chunks)))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [self.chunks[i] for i in indices]


# Build at startup
_chunks   = _chunk_text(KB_PATH.read_text(encoding="utf-8"))
rag_store = NumpyVectorStore(_chunks)

# ════════════════════════════════════════════════════════════════════════════
#  LLM — Gemini Flash with RAG context
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are Alex, a friendly and knowledgeable sales assistant for the company
    described in the context below. Your goals:

    1. Understand the customer's needs through natural conversation.
    2. Recommend the most suitable product or service from the knowledge base.
    3. Handle objections with empathy and accurate information.
    4. Keep answers concise (2-4 sentences max) since they will be spoken aloud.
    5. Never invent facts — if unsure, say you will follow up.

    CONTEXT (company knowledge base):
    {context}
""")


def llm_reply(user_text: str, history: list[dict]) -> str:
    context = "\n\n---\n\n".join(rag_store.retrieve(user_text))
    system  = SYSTEM_PROMPT.format(context=context)

    # Build contents list for the multi-turn API
    contents: list = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=turn["content"])]))
    contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=user_text)]))

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(system_instruction=system),
    )
    return response.text.strip()


# ════════════════════════════════════════════════════════════════════════════
#  STT / TTS (Google Cloud with graceful fallback)
# ════════════════════════════════════════════════════════════════════════════

def stt_transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    try:
        from google.cloud import speech_v1 as speech
        c    = speech.SpeechClient()
        cfg  = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        resp = c.recognize(config=cfg, audio=speech.RecognitionAudio(content=audio_bytes))
        return " ".join(r.alternatives[0].transcript for r in resp.results)
    except Exception as exc:
        print(f"[STT] unavailable ({exc})")
        return "[STT unavailable — use the Text Chat tab]"


def tts_synthesize(text: str) -> bytes:
    try:
        from google.cloud import texttospeech as tts
        c    = tts.TextToSpeechClient()
        resp = c.synthesize_speech(
            input=tts.SynthesisInput(text=text),
            voice=tts.VoiceSelectionParams(
                language_code="en-US", name="en-US-Neural2-D",
                ssml_gender=tts.SsmlVoiceGender.MALE,
            ),
            audio_config=tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.LINEAR16,
                sample_rate_hertz=24000,
            ),
        )
        return resp.audio_content
    except Exception as exc:
        print(f"[TTS] unavailable ({exc}); returning empty audio")
        return b""


# ════════════════════════════════════════════════════════════════════════════
#  API schemas & routes
# ════════════════════════════════════════════════════════════════════════════

class TextRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class VoiceRequest(BaseModel):
    audio_b64: str
    sample_rate: int = 16000
    history: list[dict[str, str]] = []


class AgentResponse(BaseModel):
    reply_text: str
    reply_audio_b64: str
    history: list[dict[str, str]]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "kb_chunks": len(rag_store.chunks)}


@app.post("/chat/text", response_model=AgentResponse)
def chat_text(req: TextRequest) -> AgentResponse:
    reply       = llm_reply(req.message, req.history)
    audio_bytes = tts_synthesize(reply)
    new_history = req.history + [
        {"role": "user",  "content": req.message},
        {"role": "model", "content": reply},
    ]
    return AgentResponse(
        reply_text=reply,
        reply_audio_b64=base64.b64encode(audio_bytes).decode() if audio_bytes else "",
        history=new_history,
    )


@app.post("/chat/voice", response_model=AgentResponse)
def chat_voice(req: VoiceRequest) -> AgentResponse:
    audio_bytes = base64.b64decode(req.audio_b64)
    user_text   = stt_transcribe(audio_bytes, req.sample_rate)
    if not user_text.strip():
        raise HTTPException(422, "Could not transcribe audio.")
    reply       = llm_reply(user_text, req.history)
    reply_audio = tts_synthesize(reply)
    new_history = req.history + [
        {"role": "user",  "content": user_text},
        {"role": "model", "content": reply},
    ]
    return AgentResponse(
        reply_text=reply,
        reply_audio_b64=base64.b64encode(reply_audio).decode() if reply_audio else "",
        history=new_history,
    )


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
