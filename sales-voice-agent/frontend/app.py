"""
frontend/app.py
===============
Sales Voice Agent — Gradio 6 UI

Features:
1. 🎙️ Voice Chat & 💬 Text Chat with live transcript
2. 📊 Live Lead Panel (Score, Sentiment, Intent, Stage) next to conversation
3. 📅 Automated Appointment Booking modal/form
4. 🗄️ CRM Dashboard tab (Leads table & Appointments table)
5. 📝 End-of-Call Summary popup/panel with pain points, objections & next best action
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
SAMPLE_RATE = 16_000


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _pcm_to_wav(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        pcm_int16 = (pcm * 32767).astype(np.int16)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _audio_b64_to_gradio(b64: str) -> tuple[int, np.ndarray] | None:
    if not b64:
        return None
    audio_bytes = base64.b64decode(b64)
    buf = io.BytesIO(audio_bytes)

    # 1. WAV
    try:
        with wave.open(buf, "rb") as wf:
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        return rate, samples
    except Exception:
        pass

    # 2. MP3 (ElevenLabs) via pydub
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        samples = np.array(seg.get_array_of_samples()).astype(np.float32) / 32767.0
        return seg.frame_rate, samples
    except Exception as exc:
        print(f"[Frontend] Decode failed: {exc}")
        return None


def _post(endpoint: str, payload: dict) -> dict:
    with httpx.Client(timeout=45.0) as client:
        resp = client.post(f"{API_BASE}{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()


def _get(endpoint: str) -> dict:
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{API_BASE}{endpoint}")
        resp.raise_for_status()
        return resp.json()


def _format_history(history: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "role": "user" if turn["role"] == "user" else "assistant",
            "content": turn["content"],
        }
        for turn in history
    ]


def _format_lead_panel(lead: dict) -> str:
    score = lead.get("score", 50)
    sentiment = lead.get("sentiment", "Neutral")
    intent = lead.get("intent", "Evaluating Plans")
    stage = lead.get("stage", "Discovery")
    slot = lead.get("booking_slot") or "None suggested yet"

    score_color = "#34a853" if score >= 70 else "#fbbc05" if score >= 40 else "#ea4335"

    return f"""
<div style="padding: 12px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0;">
  <div style="font-size: 16px; font-weight: bold; margin-bottom: 8px;">📊 Live Lead Qualification</div>
  <div style="margin-bottom: 6px;">
    <b>Lead Score:</b> <span style="font-size: 18px; font-weight: bold; color: {score_color};">{score}/100</span>
  </div>
  <div style="margin-bottom: 6px;"><b>Sentiment:</b> {sentiment}</div>
  <div style="margin-bottom: 6px;"><b>Intent:</b> {intent}</div>
  <div style="margin-bottom: 6px;"><b>Deal Stage:</b> {stage}</div>
  <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed #ccc; font-size: 13px; color: #555;">
    <b>Detected Slot:</b> {slot}
  </div>
</div>
"""


# ════════════════════════════════════════════════════════════════════════════
#  Event Handlers
# ════════════════════════════════════════════════════════════════════════════

def handle_text(user_msg: str, history_state: list) -> tuple:
    if not user_msg.strip():
        return gr.update(), history_state, None, "", gr.update(), gr.update()

    try:
        data = _post("/chat/text", {"message": user_msg, "history": history_state})
        new_history = data["history"]
        audio = _audio_b64_to_gradio(data.get("reply_audio_b64", ""))
        lead_html = _format_lead_panel(data.get("lead", {}))
        slot = data.get("lead", {}).get("booking_slot") or ""
        slot_update = gr.update(value=slot) if slot else gr.update()
        return _format_history(new_history), new_history, audio, "", lead_html, slot_update
    except Exception as exc:
        err = f"⚠️ Backend error: {exc}"
        return _format_history(history_state), history_state, None, err, gr.update(), gr.update()


def handle_voice(audio_tuple: tuple | None, history_state: list) -> tuple:
    if audio_tuple is None:
        return _format_history(history_state), history_state, None, "No audio recorded.", gr.update(), gr.update(), ""

    sample_rate, samples = audio_tuple
    pcm = samples.astype(np.float32) / 32767.0 if samples.dtype == np.int16 else samples
    wav_bytes = _pcm_to_wav(pcm, rate=sample_rate)
    audio_b64 = base64.b64encode(wav_bytes).decode()

    try:
        data = _post(
            "/chat/voice",
            {"audio_b64": audio_b64, "sample_rate": sample_rate, "history": history_state},
        )
        new_history = data["history"]
        audio = _audio_b64_to_gradio(data.get("reply_audio_b64", ""))
        transcript = data.get("user_transcript", "")
        lead_html = _format_lead_panel(data.get("lead", {}))
        slot = data.get("lead", {}).get("booking_slot") or ""
        slot_update = gr.update(value=slot) if slot else gr.update()
        return _format_history(new_history), new_history, audio, "", lead_html, slot_update, transcript
    except Exception as exc:
        err = f"⚠️ Backend error: {exc}"
        return _format_history(history_state), history_state, None, err, gr.update(), gr.update(), ""


def handle_book(name: str, email: str, slot: str, notes: str) -> tuple[str, list, list]:
    if not name.strip() or not email.strip() or not slot.strip():
        return "⚠️ Please fill in Name, Email, and Slot.", [], []
    try:
        res = _post("/appointments/book", {"name": name, "email": email, "slot": slot, "notes": notes})
        crm_data = _get("/crm/data")
        leads = [list(d.values()) for d in crm_data.get("leads", [])]
        appts = [list(d.values()) for d in crm_data.get("appointments", [])]
        return f"✅ {res['message']} (ID: {res['appointment_id']})", leads, appts
    except Exception as exc:
        return f"⚠️ Booking failed: {exc}", [], []


def handle_refresh_crm() -> tuple[list, list]:
    try:
        crm_data = _get("/crm/data")
        leads = [list(d.values()) for d in crm_data.get("leads", [])]
        appts = [list(d.values()) for d in crm_data.get("appointments", [])]
        return leads, appts
    except Exception:
        return [], []


def handle_end_call(history_state: list) -> str:
    if not history_state:
        return "No conversation history to summarize yet."
    try:
        res = _post("/call/summary", {"history": history_state})
        s = res.get("summary", {})
        pain_points = "".join(f"<li>{p}</li>" for p in s.get("key_pain_points", []))
        objections = "".join(f"<li>{o}</li>" for o in s.get("objections_raised", []))

        return f"""
<div style="padding: 16px; background: #eef3fc; border-radius: 8px; border: 1px solid #4285f4;">
  <h3 style="margin-top: 0; color: #1a73e8;">📞 Call Summary & Insights</h3>
  <p><b>Executive Summary:</b> {s.get('executive_summary')}</p>
  <p><b>Recommended Plan:</b> <span style="color: #0d652d; font-weight: bold;">{s.get('recommended_plan')}</span></p>
  <p><b>Key Pain Points:</b></p><ul>{pain_points}</ul>
  <p><b>Objections Raised:</b></p><ul>{objections}</ul>
  <p><b>Next Best Action:</b> <span style="font-weight: bold; color: #d93025;">{s.get('next_best_action')}</span></p>
</div>
"""
    except Exception as exc:
        return f"⚠️ Failed to summarize call: {exc}"


def clear_conversation() -> tuple:
    default_lead_html = _format_lead_panel({"score": 50, "sentiment": "Neutral", "intent": "New Inquiry", "stage": "Discovery"})
    return [], [], None, "", default_lead_html, "", ""


# ════════════════════════════════════════════════════════════════════════════
#  UI Layout
# ════════════════════════════════════════════════════════════════════════════

CSS = """
#chatbot { height: 440px; overflow-y: auto; }
.send-btn { background: #4285f4 !important; color: white !important; }
.clear-btn { background: #ea4335 !important; color: white !important; }
.book-btn { background: #34a853 !important; color: white !important; font-weight: bold !important; }
.end-btn { background: #f2994a !important; color: white !important; font-weight: bold !important; }
"""

with gr.Blocks(title="🤖 Sales Voice Agent + CRM") as demo:

    history_state = gr.State([])

    gr.Markdown(
        """
        # 🤖 Autonomous AI Sales Voice Agent
        **Powered by Gemini 3.5, Deepgram Nova-2 STT & ElevenLabs TTS**
        *End-to-End Voice • Live Lead Scoring • Automated Booking • SQLite CRM • Call Summaries*
        """
    )

    with gr.Tabs():

        # ── Tab 1: Live Voice & Chat Interaction ──────────────────────────
        with gr.TabItem("🎙️ Voice & Sales Chat"):
            with gr.Row():
                # Left Column: Conversation
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(elem_id="chatbot", label="Conversation with Alex (Sales Rep)")
                    
                    with gr.Tabs():
                        with gr.TabItem("🎤 Voice Interaction"):
                            gr.Markdown("*Speak into your microphone. Deepgram transcribes → Gemini RAG answers → ElevenLabs speaks!*")
                            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Hold to Record Voice")
                            voice_transcript_display = gr.Textbox(label="Deepgram STT Transcript", interactive=False)
                        
                        with gr.TabItem("💬 Text Interaction"):
                            with gr.Row():
                                text_in = gr.Textbox(
                                    placeholder="e.g. Which plan is best for 15 sales reps, and what about security?",
                                    label="Your Message",
                                    scale=5,
                                )
                                send_btn = gr.Button("Send ➤", elem_classes="send-btn", scale=1)

                    error_box = gr.Markdown("")

                # Right Column: Live Lead Panel & Booking & Audio Playback
                with gr.Column(scale=2):
                    audio_out = gr.Audio(label="🔊 Agent Voice Response (ElevenLabs)", autoplay=True, type="numpy")
                    lead_panel = gr.HTML(_format_lead_panel({"score": 50, "sentiment": "Neutral", "intent": "Evaluating Plans", "stage": "Discovery"}))
                    
                    gr.Markdown("### 📅 Book Sales Meeting / Demo")
                    book_name = gr.Textbox(label="Prospect Name", placeholder="e.g. John Doe")
                    book_email = gr.Textbox(label="Email", placeholder="e.g. john@acme.com")
                    book_slot = gr.Textbox(label="Confirmed Slot", placeholder="e.g. Tomorrow at 3:00 PM")
                    book_notes = gr.Textbox(label="Notes / Plan Interest", placeholder="e.g. Interested in Growth Plan free trial")
                    
                    with gr.Row():
                        book_btn = gr.Button("📅 Confirm Booking", elem_classes="book-btn")
                        end_call_btn = gr.Button("🛑 End Call & Summary", elem_classes="end-btn")
                        clear_btn = gr.Button("🗑️ Clear Chat", elem_classes="clear-btn")
                    
                    booking_status = gr.Markdown("")

            call_summary_box = gr.HTML("")

        # ── Tab 2: CRM & Pipeline Dashboard ──────────────────────────────
        with gr.TabItem("🗄️ CRM & Pipeline"):
            gr.Markdown("### 📋 Captured Leads & Scheduled Appointments (SQLite CRM)")
            refresh_crm_btn = gr.Button("🔄 Refresh CRM Data")
            
            gr.Markdown("#### 👥 Captured Leads")
            leads_table = gr.Dataframe(
                headers=["ID", "Name", "Email", "Interest", "Score", "Sentiment", "Intent", "Summary", "Created At"],
                label="Leads Pipeline",
                interactive=False,
            )
            
            gr.Markdown("#### 📅 Scheduled Appointments")
            appts_table = gr.Dataframe(
                headers=["ID", "Lead ID", "Name", "Email", "Slot", "Notes", "Status", "Created At"],
                label="Appointments",
                interactive=False,
            )

        # ── Tab 3: Architecture & About ──────────────────────────────────
        with gr.TabItem("ℹ️ Architecture & Flow"):
            gr.Markdown(
                """
                ### 🏗️ Complete End-to-End Sales Pipeline

                ```
                🎙️ Microphone (Prospect)
                        ↓
                 Deepgram Nova-2 STT (Sub-300ms transcription)
                        ↓
                 RAG Engine (NumPy Cosine Similarity on company.md)
                        ↓
                 Gemini 3.5 Flash-Lite (Sales persona + BANT qualification)
                        ↓
                 ElevenLabs TTS (Rachel Voice / Multilingual v2)
                        ↓
                 🔊 Speaker Response + 📊 Live Lead Panel + 🗄️ SQLite CRM
                ```

                ### ✨ Assignment Deliverables Covered:
                - **Voice End-to-End:** Deepgram Nova-2 + ElevenLabs TTS + audio playback + live transcript.
                - **Lead Panel:** Real-time sentiment, purchase intent, deal stage, and score (0-100).
                - **Appointment Booking:** Instant slot confirmation stored in CRM database.
                - **CRM Dashboard:** Complete searchable table of leads & appointments.
                - **End Call Summary:** Executive summary, key pain points, objections, and next best action.
                """
            )

    # ── Bind Actions ───────────────────────────────────────────────────────
    # Text Send
    send_btn.click(
        handle_text,
        inputs=[text_in, history_state],
        outputs=[chatbot, history_state, audio_out, error_box, lead_panel, book_slot],
    ).then(lambda: "", outputs=text_in)

    text_in.submit(
        handle_text,
        inputs=[text_in, history_state],
        outputs=[chatbot, history_state, audio_out, error_box, lead_panel, book_slot],
    ).then(lambda: "", outputs=text_in)

    # Voice Input
    audio_in.stop_recording(
        handle_voice,
        inputs=[audio_in, history_state],
        outputs=[chatbot, history_state, audio_out, error_box, lead_panel, book_slot, voice_transcript_display],
    )

    # Booking
    book_btn.click(
        handle_book,
        inputs=[book_name, book_email, book_slot, book_notes],
        outputs=[booking_status, leads_table, appts_table],
    )

    # CRM Refresh
    refresh_crm_btn.click(
        handle_refresh_crm,
        outputs=[leads_table, appts_table],
    )

    # End Call Summary
    end_call_btn.click(
        handle_end_call,
        inputs=[history_state],
        outputs=[call_summary_box],
    )

    # Clear
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, history_state, audio_out, error_box, lead_panel, book_slot, call_summary_box],
    )


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
