"""
backend/main.py
===============
Sales Voice Agent — FastAPI backend

Pipeline
--------
User audio (base64)
  → Speech-to-Text  (Google Cloud Speech / Gemini Live)
  → RAG retrieval   (FAISS + Gemini embeddings)
  → LLM answer      (Gemini Flash)
  → Text-to-Speech  (Google Cloud TTS)
  → audio response  (base64 WAV)
"""

from __future__ import annotations

import base64
import io
import json
import os
import textwrap
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Google Generative AI ────────────────────────────────────────────────────
import google.generativeai as genai
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech

# ── Load env ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in .env")

genai.configure(api_key=GEMINI_API_KEY)

# ── Constants ───────────────────────────────────────────────────────────────
KB_PATH = Path(__file__).parent.parent / "kb" / "company.md"
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-1.5-flash-latest"
CHUNK_SIZE = 400          # characters per RAG chunk
CHUNK_OVERLAP = 80
TOP_K = 4                 # chunks retrieved per query
EMBED_DIM = 768           # text-embedding-004 output dimension

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="Sales Voice Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
#  RAG — build FAISS index from knowledge base at startup
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping character chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _embed(texts: list[str]) -> np.ndarray:
    """Return (N, EMBED_DIM) float32 matrix via Gemini embeddings."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    return np.array(result["embedding"], dtype=np.float32)


class RAGIndex:
    """Thin wrapper around a FAISS flat index + raw chunk store."""

    def __init__(self, kb_path: Path) -> None:
        raw = kb_path.read_text(encoding="utf-8")
        self.chunks = _chunk_text(raw)
        embeddings = _embed(self.chunks)
        # L2-normalise → cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.index.add(embeddings)
        print(f"[RAG] Indexed {len(self.chunks)} chunks from {kb_path.name}")

    def retrieve(self, query: str, k: int = TOP_K) -> list[str]:
        q_emb = _embed([query])
        faiss.normalize_L2(q_emb)
        _, ids = self.index.search(q_emb, k)
        return [self.chunks[i] for i in ids[0] if i != -1]


# Build index once at import time (FastAPI startup)
rag_index = RAGIndex(KB_PATH)

# ═══════════════════════════════════════════════════════════════════════════
#  Gemini chat session (stateless — history passed per request)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are Alex, a friendly and knowledgeable sales assistant for the company
    described in the context below. Your goal is to:

    1. Understand the customer's needs through natural conversation.
    2. Recommend the most suitable product or service from the knowledge base.
    3. Handle objections with empathy and accurate information.
    4. Keep answers concise (2-4 sentences max) since they will be spoken aloud.
    5. Never invent facts — if unsure, say you will follow up.

    CONTEXT (company knowledge base):
    {context}
""")


def _build_prompt(context: str, history: list[dict], user_message: str) -> list[dict]:
    """Assemble the message list for Gemini (system injected as first user turn)."""
    messages: list[dict] = [
        {"role": "user", "parts": [SYSTEM_PROMPT.format(context=context)]},
        {"role": "model", "parts": ["Understood. I'm ready to help customers!"]},
    ]
    for turn in history:
        messages.append({"role": turn["role"], "parts": [turn["content"]]})
    messages.append({"role": "user", "parts": [user_message]})
    return messages


def llm_reply(user_text: str, history: list[dict]) -> str:
    """RAG-augmented Gemini answer."""
    context_chunks = rag_index.retrieve(user_text)
    context = "\n\n---\n\n".join(context_chunks)
    messages = _build_prompt(context, history, user_text)
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(messages)
    return response.text.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Speech helpers (Google Cloud — credentials via GOOGLE_APPLICATION_CREDENTIALS)
#  Falls back to a stub if credentials are absent (dev mode).
# ═══════════════════════════════════════════════════════════════════════════

def stt_transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe raw LINEAR16 audio bytes → text."""
    try:
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        resp = client.recognize(config=config, audio=audio)
        return " ".join(r.alternatives[0].transcript for r in resp.results)
    except Exception as exc:  # noqa: BLE001
        print(f"[STT] Cloud unavailable ({exc}); returning placeholder.")
        return "[STT unavailable — type your message instead]"


def tts_synthesize(text: str) -> bytes:
    """Synthesize *text* → WAV bytes (LINEAR16, 24 kHz)."""
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-D",          # natural male voice
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as exc:  # noqa: BLE001
        print(f"[TTS] Cloud unavailable ({exc}); returning empty audio.")
        return b""


# ═══════════════════════════════════════════════════════════════════════════
#  API schemas
# ═══════════════════════════════════════════════════════════════════════════

class TextRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []   # [{"role": "user"|"model", "content": "..."}]


class VoiceRequest(BaseModel):
    audio_b64: str                        # base64-encoded LINEAR16 PCM
    sample_rate: int = 16000
    history: list[dict[str, str]] = []


class AgentResponse(BaseModel):
    reply_text: str
    reply_audio_b64: str                  # base64 WAV (empty string if TTS unavailable)
    history: list[dict[str, str]]


# ═══════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "kb_chunks": len(rag_index.chunks)}


@app.post("/chat/text", response_model=AgentResponse)
def chat_text(req: TextRequest) -> AgentResponse:
    """Pure-text round-trip (no audio)."""
    reply = llm_reply(req.message, req.history)
    audio_bytes = tts_synthesize(reply)
    new_history = req.history + [
        {"role": "user", "content": req.message},
        {"role": "model", "content": reply},
    ]
    return AgentResponse(
        reply_text=reply,
        reply_audio_b64=base64.b64encode(audio_bytes).decode() if audio_bytes else "",
        history=new_history,
    )


@app.post("/chat/voice", response_model=AgentResponse)
def chat_voice(req: VoiceRequest) -> AgentResponse:
    """Voice round-trip: audio-in → text → LLM → audio-out."""
    audio_bytes = base64.b64decode(req.audio_b64)
    user_text = stt_transcribe(audio_bytes, req.sample_rate)
    if not user_text.strip():
        raise HTTPException(status_code=422, detail="Could not transcribe audio.")
    reply = llm_reply(user_text, req.history)
    reply_audio = tts_synthesize(reply)
    new_history = req.history + [
        {"role": "user", "content": user_text},
        {"role": "model", "content": reply},
    ]
    return AgentResponse(
        reply_text=reply,
        reply_audio_b64=base64.b64encode(reply_audio).decode() if reply_audio else "",
        history=new_history,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
