"""
frontend/app.py
===============
Sales Voice Agent — Gradio UI

Tabs
----
🎙 Voice Chat   — record microphone → agent speaks back
💬 Text Chat    — type a message  → read & hear the reply
ℹ️  About        — quick guide
"""

from __future__ import annotations

import base64
import io
import json
import os
import wave
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SAMPLE_RATE = 16_000   # Hz — must match backend STT config


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _pcm_to_wav(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 numpy array to raw WAV bytes (LINEAR16)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(rate)
        pcm_int16 = (pcm * 32767).astype(np.int16)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _audio_b64_to_gradio(b64: str) -> tuple[int, np.ndarray] | None:
    """Decode a base64 WAV string into a Gradio-compatible (rate, ndarray)."""
    if not b64:
        return None
    wav_bytes = base64.b64decode(b64)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return rate, samples


def _post(endpoint: str, payload: dict) -> dict:
    """POST to the FastAPI backend; raise on error."""
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_BASE}{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()


def _format_history(history: list[dict]) -> list[tuple[str, str]]:
    """Convert flat history list → Gradio chatbot format [(user, bot), ...]."""
    pairs: list[tuple[str, str]] = []
    it = iter(history)
    for turn in it:
        if turn["role"] == "user":
            try:
                bot = next(it)
                pairs.append((turn["content"], bot["content"]))
            except StopIteration:
                pairs.append((turn["content"], "…"))
    return pairs


# ════════════════════════════════════════════════════════════════════════════
#  Gradio event handlers
# ════════════════════════════════════════════════════════════════════════════

def handle_text(user_msg: str, history_state: list) -> tuple:
    """Text-chat handler."""
    if not user_msg.strip():
        return gr.update(), history_state, None, ""

    try:
        data = _post("/chat/text", {"message": user_msg, "history": history_state})
    except Exception as exc:  # noqa: BLE001
        err = f"⚠️ Backend error: {exc}"
        return _format_history(history_state), history_state, None, err

    new_history = data["history"]
    audio = _audio_b64_to_gradio(data.get("reply_audio_b64", ""))
    return _format_history(new_history), new_history, audio, ""


def handle_voice(audio_tuple: tuple | None, history_state: list) -> tuple:
    """Voice-chat handler: microphone audio → agent response."""
    if audio_tuple is None:
        return _format_history(history_state), history_state, None, "No audio captured."

    sample_rate, samples = audio_tuple
    wav_bytes = _pcm_to_wav(samples.astype(np.float32) / 32767.0
                            if samples.dtype == np.int16 else samples,
                            rate=sample_rate)
    audio_b64 = base64.b64encode(wav_bytes).decode()

    try:
        data = _post(
            "/chat/voice",
            {"audio_b64": audio_b64, "sample_rate": sample_rate, "history": history_state},
        )
    except Exception as exc:  # noqa: BLE001
        err = f"⚠️ Backend error: {exc}"
        return _format_history(history_state), history_state, None, err

    new_history = data["history"]
    audio = _audio_b64_to_gradio(data.get("reply_audio_b64", ""))
    return _format_history(new_history), new_history, audio, ""


def clear_conversation() -> tuple:
    return [], [], None, ""


# ════════════════════════════════════════════════════════════════════════════
#  UI layout
# ════════════════════════════════════════════════════════════════════════════

CSS = """
#chatbot { height: 480px; overflow-y: auto; }
.send-btn { background: #4285f4 !important; color: white !important; }
.clear-btn { background: #ea4335 !important; color: white !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="🤖 Sales Voice Agent") as demo:

    # ── shared state ──────────────────────────────────────────────────────
    history_state = gr.State([])    # list[{"role": ..., "content": ...}]

    # ── header ────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🤖 Sales Voice Agent
        **Powered by Gemini + RAG** — Ask about products, pricing, or anything else!
        """
    )

    with gr.Tabs():

        # ── Tab 1: Voice Chat ─────────────────────────────────────────────
        with gr.TabItem("🎙️ Voice Chat"):
            gr.Markdown("*Record your question — the agent will respond in voice and text.*")

            with gr.Row():
                with gr.Column(scale=3):
                    voice_chatbot = gr.Chatbot(elem_id="chatbot", label="Conversation")
                    voice_audio_in = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 Hold to record",
                    )
                    voice_error = gr.Markdown("", visible=True)
                with gr.Column(scale=1):
                    voice_audio_out = gr.Audio(
                        label="🔊 Agent response",
                        autoplay=True,
                        type="numpy",
                    )
                    voice_clear = gr.Button("🗑️ Clear", elem_classes="clear-btn")

            voice_audio_in.stop_recording(
                handle_voice,
                inputs=[voice_audio_in, history_state],
                outputs=[voice_chatbot, history_state, voice_audio_out, voice_error],
            )
            voice_clear.click(
                clear_conversation,
                outputs=[voice_chatbot, history_state, voice_audio_out, voice_error],
            )

        # ── Tab 2: Text Chat ──────────────────────────────────────────────
        with gr.TabItem("💬 Text Chat"):
            gr.Markdown("*Type your question below.*")

            with gr.Row():
                with gr.Column(scale=3):
                    text_chatbot = gr.Chatbot(elem_id="chatbot", label="Conversation")
                    with gr.Row():
                        text_input = gr.Textbox(
                            placeholder="e.g. What's your best plan for a small business?",
                            label="Your message",
                            scale=5,
                            lines=1,
                        )
                        text_send = gr.Button("Send ➤", elem_classes="send-btn", scale=1)
                    text_error = gr.Markdown("")
                with gr.Column(scale=1):
                    text_audio_out = gr.Audio(
                        label="🔊 Agent response",
                        autoplay=True,
                        type="numpy",
                    )
                    text_clear = gr.Button("🗑️ Clear", elem_classes="clear-btn")

            text_send.click(
                handle_text,
                inputs=[text_input, history_state],
                outputs=[text_chatbot, history_state, text_audio_out, text_error],
            ).then(lambda: "", outputs=text_input)

            text_input.submit(
                handle_text,
                inputs=[text_input, history_state],
                outputs=[text_chatbot, history_state, text_audio_out, text_error],
            ).then(lambda: "", outputs=text_input)

            text_clear.click(
                clear_conversation,
                outputs=[text_chatbot, history_state, text_audio_out, text_error],
            )

        # ── Tab 3: About ──────────────────────────────────────────────────
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(
                """
                ## How it works

                ```
                Microphone / Text
                      ↓
                  Speech-to-Text  (Google Cloud Speech)
                      ↓
                  RAG Retrieval   (FAISS + Gemini Embeddings)
                      ↓
                  LLM Answer      (Gemini 1.5 Flash)
                      ↓
                  Text-to-Speech  (Google Cloud TTS)
                      ↓
                 Speaker / Chat
                ```

                ## Quick start

                ```bash
                # 1. Install deps
                pip install -r requirements.txt

                # 2. Set your key in .env
                GEMINI_API_KEY=your_key_here

                # 3. Start backend
                python -m backend.main

                # 4. Start frontend (this file)
                python frontend/app.py
                ```

                ## Stack
                | Layer | Technology |
                |-------|-----------|
                | LLM | Gemini 1.5 Flash |
                | Embeddings | text-embedding-004 |
                | Vector DB | FAISS (local) |
                | STT | Google Cloud Speech |
                | TTS | Google Cloud TTS Neural2 |
                | API | FastAPI + Uvicorn |
                | UI | Gradio 4 |
                """
            )


# ════════════════════════════════════════════════════════════════════════════
#  Launch
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
