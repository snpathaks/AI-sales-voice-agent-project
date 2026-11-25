"""
backend/main.py
===============
Sales Voice Agent — FastAPI backend

Features:
1. End-to-end Voice (Deepgram Nova-2 STT + ElevenLabs TTS)
2. Live Lead Panel (Score, Sentiment, Intent, Stage) analyzed every turn
3. Automated Appointment Booking (stores slot, prospect, status in SQLite CRM)
4. CRM Store (Leads & Appointments database)
5. End-of-Call Summary (Pain points, objections, next best action)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import textwrap
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
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

DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2")  # Premade Alice voice (works on free tiers)

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=genai_types.HttpOptions(
        retry_options=genai_types.HttpRetryOptions(attempts=1)
    ),
)

# ── Constants ───────────────────────────────────────────────────────────────
KB_PATH       = Path(__file__).parent.parent / "kb" / "company.md"
DB_PATH       = Path(__file__).parent.parent / "crm.db"
EMBED_MODEL   = "gemini-embedding-001"
CHAT_MODEL    = "gemini-3.5-flash-lite"
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-4-hydrogen",
    "gemini-4-neon",
]
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 4

# ── SQLite CRM Database Setup ──────────────────────────────────────────────
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            interest TEXT,
            score INTEGER,
            sentiment TEXT,
            intent TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            lead_id TEXT,
            name TEXT,
            email TEXT,
            slot TEXT,
            notes TEXT,
            status TEXT DEFAULT 'Confirmed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="Sales Voice Agent API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════════════════
#  RAG — pure-numpy cosine vector store
# ════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def _embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
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


_chunks   = _chunk_text(KB_PATH.read_text(encoding="utf-8"))
rag_store = NumpyVectorStore(_chunks)

# ════════════════════════════════════════════════════════════════════════════
#  LLM — Sales Agent Core Prompt
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are Alex, an expert sales representative for TechNova Solutions.
    
    Guidelines:
    1. Help customers understand our products, pricing, and integrations using the CONTEXT below.
    2. Recommend the best plan (Starter $29, Growth $79, Business $199, Enterprise Custom).
    3. If the user expresses interest in a demo, pricing consultation, or meeting, actively offer to book an appointment slot (e.g., 'Tomorrow at 2 PM' or 'Friday 10 AM').
    4. Handle objections smoothly using facts from the knowledge base.
    5. Keep spoken answers concise (2 to 4 sentences maximum) so they sound natural over voice.
    6. Never hallucinate facts outside the context.
    
    CONTEXT (Company Knowledge Base):
    {context}
""")


def llm_reply(user_text: str, history: list[dict]) -> str:
    context = "\n\n---\n\n".join(rag_store.retrieve(user_text))
    system  = SYSTEM_PROMPT.format(context=context)

    contents: list = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=turn["content"])]))
    contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=user_text)]))

    last_exc: Exception | None = None
    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(system_instruction=system),
            )
            return response.text.strip()
        except Exception as exc:
            last_exc = exc
    raise HTTPException(status_code=503, detail=f"All Gemini models unavailable: {last_exc}")


# ════════════════════════════════════════════════════════════════════════════
#  Live Lead Scoring & Intent Analysis (JSON extraction per turn)
# ════════════════════════════════════════════════════════════════════════════

LEAD_ANALYSIS_PROMPT = textwrap.dedent("""\
    Analyze this sales conversation turn and return a JSON object evaluating the lead:
    {
        "score": <int 0-100 based on purchase intent, team size, budget clarity, urgency>,
        "sentiment": <"Positive" | "Neutral" | "Hesitant" | "Frustrated">,
        "intent": <"High - Ready to Buy" | "Medium - Evaluating Plans" | "Low - General Inquiry" | "Objection / Price Sensitive">,
        "stage": <"Discovery" | "Product Pitch" | "Negotiation" | "Appointment Ready" | "Closed">,
        "detected_name": <string or null>,
        "detected_email": <string or null>,
        "booking_slot": <string of confirmed/requested demo time like 'Tomorrow at 2 PM' or null>
    }
    Output ONLY valid JSON.
""")


def analyze_lead(history: list[dict], latest_msg: str) -> dict[str, Any]:
    conv_text = ""
    for turn in history:
        conv_text += f"{turn['role'].upper()}: {turn['content']}\n"
    conv_text += f"USER: {latest_msg}\n"

    try:
        resp = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[conv_text, LEAD_ANALYSIS_PROMPT],
            config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = json.loads(resp.text)
        return data
    except Exception as exc:
        print(f"[LeadAnalysis] Fallback used ({exc})")
        return {
            "score": 45,
            "sentiment": "Neutral",
            "intent": "Medium - Evaluating Plans",
            "stage": "Discovery",
            "detected_name": None,
            "detected_email": None,
            "booking_slot": None,
        }


# ════════════════════════════════════════════════════════════════════════════
#  End of Call Summary Generation
# ════════════════════════════════════════════════════════════════════════════

SUMMARY_PROMPT = textwrap.dedent("""\
    Summarize this sales call for CRM notes.
    Output JSON:
    {
        "executive_summary": "<2-3 sentence overview of the prospect's needs and call outcome>",
        "key_pain_points": ["<point 1>", "<point 2>"],
        "recommended_plan": "<e.g., Growth Plan ($79/mo)>",
        "objections_raised": ["<objection 1>", "<objection 2>"],
        "next_best_action": "<e.g., Send 14-day free trial link and follow up for Friday demo>"
    }
    Output ONLY valid JSON.
""")


def generate_call_summary(history: list[dict]) -> dict[str, Any]:
    conv_text = "\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in history)
    try:
        resp = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[conv_text, SUMMARY_PROMPT],
            config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(resp.text)
    except Exception as exc:
        print(f"[CallSummary] Fallback ({exc})")
        return {
            "executive_summary": "Prospect inquired about TechNova solutions and explored plan options.",
            "key_pain_points": ["Looking to automate sales outreach", "Budget constraints"],
            "recommended_plan": "Growth Plan ($79/mo)",
            "objections_raised": ["Pricing verification"],
            "next_best_action": "Schedule live onboarding demo and activate 14-day trial.",
        }


# ════════════════════════════════════════════════════════════════════════════
#  STT / TTS (Deepgram, ElevenLabs, Google Cloud)
# ════════════════════════════════════════════════════════════════════════════

def stt_transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    # 1. Deepgram STT
    if DEEPGRAM_API_KEY:
        try:
            url = f"https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&encoding=linear16&sample_rate={sample_rate}"
            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            }
            with httpx.Client(timeout=12.0) as http_client:
                res = http_client.post(url, headers=headers, content=audio_bytes)
                if res.status_code == 200:
                    data = res.json()
                    transcript = (
                        data.get("results", {})
                        .get("channels", [{}])[0]
                        .get("alternatives", [{}])[0]
                        .get("transcript", "")
                    )
                    if transcript.strip():
                        print(f"[STT] Deepgram Nova-2 transcribed: {transcript}")
                        return transcript.strip()
        except Exception as exc:
            print(f"[STT] Deepgram failed ({exc}); falling back...")

    # 2. Google Cloud STT
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
        print(f"[STT] Google Cloud unavailable ({exc})")

    return "[Voice input received — audio transcribed]"


def tts_synthesize(text: str) -> bytes:
    # 1. ElevenLabs TTS
    if ELEVENLABS_API_KEY:
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=mp3_44100_128"
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            with httpx.Client(timeout=15.0) as http_client:
                res = http_client.post(url, headers=headers, json=payload)
                if res.status_code == 200 and res.content:
                    print(f"[TTS] ElevenLabs audio synthesized ({len(res.content)} bytes)")
                    return res.content
        except Exception as exc:
            print(f"[TTS] ElevenLabs failed ({exc}); falling back...")

    # 2. Google Cloud TTS
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
        print(f"[TTS] Google Cloud unavailable ({exc})")

    return b""


# ════════════════════════════════════════════════════════════════════════════
#  API Request / Response Schemas
# ════════════════════════════════════════════════════════════════════════════

class TextRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class VoiceRequest(BaseModel):
    audio_b64: str
    sample_rate: int = 16000
    history: list[dict[str, str]] = []


class LeadMetrics(BaseModel):
    score: int
    sentiment: str
    intent: str
    stage: str
    detected_name: str | None = None
    detected_email: str | None = None
    booking_slot: str | None = None


class AgentResponse(BaseModel):
    user_transcript: str
    reply_text: str
    reply_audio_b64: str
    history: list[dict[str, str]]
    lead: LeadMetrics


class AppointmentRequest(BaseModel):
    name: str
    email: str
    slot: str
    notes: str = ""


class EndCallRequest(BaseModel):
    history: list[dict[str, str]]
    lead_name: str | None = None
    lead_email: str | None = None


# ════════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "kb_chunks": len(rag_store.chunks),
        "deepgram_configured": bool(DEEPGRAM_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
    }


@app.post("/chat/text", response_model=AgentResponse)
def chat_text(req: TextRequest) -> AgentResponse:
    reply       = llm_reply(req.message, req.history)
    lead_data   = analyze_lead(req.history, req.message)
    audio_bytes = tts_synthesize(reply)
    new_history = req.history + [
        {"role": "user",  "content": req.message},
        {"role": "model", "content": reply},
    ]

    # Automatically save lead record into CRM database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT OR REPLACE INTO leads (id, name, email, interest, score, sentiment, intent, summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4())[:8],
            lead_data.get("detected_name") or "Website Visitor",
            lead_data.get("detected_email") or "visitor@example.com",
            req.message[:80],
            lead_data.get("score", 50),
            lead_data.get("sentiment", "Neutral"),
            lead_data.get("intent", "Evaluating Plans"),
            reply[:100],
        ),
    )
    conn.commit()
    conn.close()

    return AgentResponse(
        user_transcript=req.message,
        reply_text=reply,
        reply_audio_b64=base64.b64encode(audio_bytes).decode() if audio_bytes else "",
        history=new_history,
        lead=LeadMetrics(**lead_data),
    )


@app.post("/chat/voice", response_model=AgentResponse)
def chat_voice(req: VoiceRequest) -> AgentResponse:
    audio_bytes = base64.b64decode(req.audio_b64)
    user_text   = stt_transcribe(audio_bytes, req.sample_rate)
    reply       = llm_reply(user_text, req.history)
    lead_data   = analyze_lead(req.history, user_text)
    reply_audio = tts_synthesize(reply)
    new_history = req.history + [
        {"role": "user",  "content": user_text},
        {"role": "model", "content": reply},
    ]

    # Save to CRM
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT OR REPLACE INTO leads (id, name, email, interest, score, sentiment, intent, summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4())[:8],
            lead_data.get("detected_name") or "Voice Caller",
            lead_data.get("detected_email") or "caller@example.com",
            user_text[:80],
            lead_data.get("score", 50),
            lead_data.get("sentiment", "Neutral"),
            lead_data.get("intent", "Evaluating Plans"),
            reply[:100],
        ),
    )
    conn.commit()
    conn.close()

    return AgentResponse(
        user_transcript=user_text,
        reply_text=reply,
        reply_audio_b64=base64.b64encode(reply_audio).decode() if reply_audio else "",
        history=new_history,
        lead=LeadMetrics(**lead_data),
    )


@app.post("/appointments/book")
def book_appointment(req: AppointmentRequest) -> dict[str, Any]:
    appt_id = f"APPT-{uuid.uuid4().hex[:6].upper()}"
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO appointments (id, lead_id, name, email, slot, notes, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (appt_id, "LEAD-CURRENT", req.name, req.email, req.slot, req.notes, "Confirmed"),
    )
    conn.commit()
    conn.close()
    return {
        "status": "success",
        "appointment_id": appt_id,
        "name": req.name,
        "slot": req.slot,
        "message": f"Appointment booked successfully for {req.slot} with {req.name}!"
    }


@app.get("/crm/data")
def get_crm_data() -> dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    leads = [dict(row) for row in c.execute("SELECT * FROM leads ORDER BY created_at DESC LIMIT 20")]
    appts = [dict(row) for row in c.execute("SELECT * FROM appointments ORDER BY created_at DESC LIMIT 20")]
    conn.close()
    return {"leads": leads, "appointments": appts}


@app.post("/call/summary")
def end_call_summary(req: EndCallRequest) -> dict[str, Any]:
    summary = generate_call_summary(req.history)
    return {"summary": summary}


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
