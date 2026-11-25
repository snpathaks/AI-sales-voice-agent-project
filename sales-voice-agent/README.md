# 🤖 Autonomous AI Sales Voice Agent + CRM Pipeline

An enterprise-grade, end-to-end **Autonomous AI Sales Voice & Chat Agent** built with **FastAPI**, **Gradio 6**, **Google Gemini**, **Deepgram Nova-2 (STT)**, **ElevenLabs (TTS)**, and an integrated **SQLite CRM & Lead Qualification Pipeline**.

The agent, named **Alex**, acts as an autonomous sales representative for **TechNova Solutions** (defined in [`kb/company.md`](kb/company.md)), answering questions via natural voice and text, scoring leads in real time, booking demos, saving data to a CRM database, and generating post-call executive summaries.

---

## 📋 Table of Contents

- [1. Executive Summary & Features](#-1-executive-summary--features)
- [2. System Architecture](#-2-system-architecture)
- [3. End-to-End Business Workflow](#-3-end-to-end-business-workflow)
- [4. Project Structure & Tech Stack](#-4-project-structure--tech-stack)
- [5. Setup & Running Locally](#-5-setup--running-locally)
- [6. Evaluation Demonstration Walkthrough](#-6-evaluation-demonstration-walkthrough)
- [7. Scalability Strategy (Up to 100,000 Users)](#-7-scalability-strategy-up-to-100000-users)
- [8. Security, Infrastructure & Operational Economics](#-8-security-infrastructure--operational-economics)

---

## 🌟 1. Executive Summary & Features

| Feature | Technology | Capabilities |
|---|---|---|
| **🎙️ End-to-End Voice (STT & TTS)** | Deepgram Nova-2 + ElevenLabs | Record voice via microphone, transcribe with sub-300ms latency, generate grounded responses, and play natural human-like audio (*Alice / Rachel* voice). Displays live user transcripts. |
| **📚 Grounded Knowledge Retrieval (RAG)** | Gemini Embeddings + Pure NumPy | 20 semantic chunks from `kb/company.md` embedded at startup via `gemini-embedding-001`. L2-normalized cosine similarity retrieves top-4 relevant chunks per turn to prevent hallucination. |
| **🧠 Resilient Multi-Model LLM** | Google GenAI SDK | Primary `gemini-3.5-flash-lite` with instant fallback chain (`gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-4-hydrogen`, `gemini-4-neon`) protecting against 503 traffic spikes. Constrained to 2–4 spoken sentences. |
| **📊 Real-Time Lead Qualification** | Gemini Structured JSON | Analyzes conversation turn-by-turn: computes **Lead Score (0–100)**, **Sentiment** (*Positive/Neutral/Hesitant*), **Purchase Intent** (*High/Medium/Low*), and **Deal Stage** (*Discovery/Negotiation/Appointment Ready*). |
| **📅 Automated Appointment Booking** | FastAPI + SQLite CRM | Automatically detects proposed meeting slots (e.g., *"Tomorrow at 2:00 PM"*), confirms prospect name and email, and generates persistent appointment IDs (`APPT-XXXXXX`). |
| **🗄️ CRM & Pipeline Dashboard Tab** | SQLite (`crm.db`) + Gradio 6 Dataframe | Two searchable, auto-updating tables for **Captured Leads** and **Scheduled Appointments**, complete with manual refresh controls. |
| **📝 End-of-Call Summary & Insights** | Gemini Structured JSON | Generates an **Executive Summary**, **Key Pain Points**, **Recommended Tier**, **Objections Raised**, and **Next Best Action** for sales reps. |

---

## 🏗️ 2. System Architecture

```mermaid
flowchart TD
    User["👤 Prospect (Web / Microphone)"] <--> UI["🖥️ Gradio 6 Frontend (Port 7860)<br/>Chat + Lead Panel + Booking + CRM Tables"]
    
    UI -->|Audio (Base64 WAV)| STT["🎙️ Deepgram Nova-2 STT<br/>Sub-300ms Speech Recognition"]
    STT --> API["⚡ FastAPI Backend (Port 8000)<br/>backend/main.py"]
    UI -->|Text Message| API

    subgraph Intelligence["AI & RAG Intelligence Engine"]
        API --> Embed["Gemini Embeddings<br/>gemini-embedding-001"]
        Embed --> VStore[("NumPy Vector Store<br/>Cosine Similarity (Top-4)")]
        KB[("📄 Knowledge Base<br/>kb/company.md")] -.->|Startup Indexing| VStore
        VStore --> Context["Context Chunks"]
        
        Context --> LLM["🧠 Gemini 3.5 Flash-Lite<br/>(Instant Multi-Model Fallback Chain)"]
        API --> LeadEngine["📊 Lead Analyzer & JSON Classifier<br/>Score, Sentiment, Intent, Stage"]
    end

    LLM --> TTS["🔊 ElevenLabs TTS<br/>Premade Alice Voice (MP3 44.1kHz)"]
    TTS --> AudioOut["Base64 Audio Response"]
    AudioOut --> UI

    LeadEngine --> CRMDB[("🗄️ SQLite Database (crm.db)<br/>Leads & Appointments Tables")]
    API -->|Book Appointment| CRMDB
    CRMDB -->|GET /crm/data| UI
```

---

## 🔄 3. End-to-End Business Workflow

1. **Inbound Inquiry:** Prospect speaks into microphone or types: *"Hi, I run a 10-person sales team and need CRM automation."*
2. **Deepgram Transcription:** Incoming audio is converted to text in ~250ms and displayed on screen.
3. **RAG Context Retrieval:** NumPy cosine engine queries `kb/company.md` chunks, extracting the **Growth Plan (\$79/mo, 15 users, 300 Voice AI mins)**.
4. **Sales Pitch & Spoken Output:** Gemini responds as **Alex** in 2–3 concise sentences, offering a 14-day trial and a demo slot. ElevenLabs synthesizes spoken audio and plays it automatically in the UI.
5. **Live Lead Scoring:** The lead panel updates instantly: Score **85/100**, Sentiment **Positive**, Intent **High - Ready to Buy**, Deal Stage **Discovery**.
6. **Slot Confirmation & Booking:** User says: *"Tomorrow at 2 PM works. My name is Alice, alice@startup.io."* The booking panel pre-fills the slot, and clicking **Confirm Booking** logs `APPT-XXXXXX` into the CRM database.
7. **End-of-Call Summary:** Rep clicks **🛑 End Call & Summary** to view structured bullet points (Executive Summary, Pain Points, Objections, Next Best Action).

---

## 📁 4. Project Structure & Tech Stack

```text
sales-voice-agent/
├── .env                  # API keys (GEMINI, DEEPGRAM, ELEVENLABS) [GIT IGNORED]
├── .env.example          # Template showing required environment variables
├── .gitignore            # Excludes secrets, databases, caches, audio files
├── requirements.txt      # Clean, Python 3.10-3.14 compatible dependencies
├── crm.db                # SQLite database for leads and appointments [GIT IGNORED]
├── README.md             # Comprehensive architecture and evaluation guide
├── backend/
│   ├── __init__.py
│   └── main.py           # FastAPI server (RAG, Gemini, Deepgram, ElevenLabs, CRM)
├── frontend/
│   └── app.py            # Gradio 6 UI (Voice/Text Chat, Lead Panel, CRM Tab, Summary)
└── kb/
    └── company.md        # TechNova Solutions knowledge base (Pricing, SLAs, FAQs)
```

| Layer | Technology |
|---|---|
| **LLM & Reasoning** | Google Gemini (`gemini-3.5-flash-lite`, `google-genai` SDK v1.x) |
| **Embeddings & RAG** | `gemini-embedding-001` + NumPy L2-normalized cosine similarity |
| **Speech-to-Text (STT)** | Deepgram Nova-2 (`https://api.deepgram.com/v1/listen`) |
| **Text-to-Speech (TTS)** | ElevenLabs (`https://api.elevenlabs.io/v1/text-to-speech`) |
| **API Framework** | FastAPI + Uvicorn + Pydantic v2 |
| **Database & CRM** | SQLite3 (`crm.db` persistent storage) |
| **Frontend UI** | Gradio 6 (`gr.Blocks`, `gr.Chatbot`, `gr.Dataframe`) |

---

## 🚀 5. Setup & Running Locally

### Prerequisites
- Python **3.10+** (fully verified on Python 3.14)
- API Keys:
  - **Gemini API Key** ([Google AI Studio](https://aistudio.google.com/))
  - **Deepgram API Key** ([Deepgram Console](https://console.deepgram.com/))
  - **ElevenLabs API Key** ([ElevenLabs](https://elevenlabs.io/))

### Installation & Launch

1. **Clone & Configure Environment:**
   ```bash
   cp .env.example .env
   # Open .env and add your GEMINI_API_KEY, DEEPGRAM_API_KEY, and ELEVENLABS_API_KEY
   ```

2. **Install Dependencies:**
   ```bash
   cd sales-voice-agent
   pip install google-genai fastapi "uvicorn[standard]" python-dotenv httpx numpy pydantic gradio pydub markdown
   ```

3. **Start the FastAPI Backend (Port 8000):**
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
   *Health Check:* `http://localhost:8000/health` → `{"status":"ok","kb_chunks":20,"deepgram_configured":true,"elevenlabs_configured":true}`  
   *Interactive Swagger Docs:* `http://localhost:8000/docs`

4. **Start the Gradio Frontend (Port 7860):**
   ```bash
   python frontend/app.py
   ```
   Open **`http://localhost:7860`** in your browser.

---

## 🎯 6. Evaluation Demonstration Walkthrough

During your technical evaluation, follow this 4-step sequence:

### Step 1: Voice & Knowledge Retrieval (RAG)
1. Go to **🎙️ Voice & Sales Chat** tab → click **🎤 Voice Interaction**.
2. Hold to record: *"Tell me about TechNova Solutions and your Growth plan."*
3. **Point out:**
   - Deepgram transcript appears immediately in the text box.
   - Alex answers in 2 sentences strictly from `company.md`.
   - ElevenLabs voice response plays aloud automatically.

### Step 2: Live Lead Scoring & Intent Analysis
1. Show the **📊 Live Lead Qualification Panel** on the right.
2. Note how **Score** (e.g. 85/100), **Sentiment** (*Positive*), **Intent** (*High - Ready to Buy*), and **Deal Stage** (*Discovery*) updated automatically.

### Step 3: Appointment Booking
1. Ask Alex: *"Can I book a demo for tomorrow at 2 PM? My name is John, john@corp.com."*
2. Note how the **Confirmed Slot** box auto-populates with *"Tomorrow at 2 PM"*.
3. Click **📅 Confirm Booking** → displays green confirmation and assigns persistent ID (`APPT-XXXXXX`).

### Step 4: CRM Pipeline & Call Summary
1. Switch to the **🗄️ CRM & Pipeline** tab → click **🔄 Refresh CRM Data** to show the live SQLite records in the **Captured Leads** and **Appointments** tables.
2. Switch back and click **🛑 End Call & Summary** → show the auto-generated **Executive Summary**, **Key Pain Points**, **Objections**, and **Next Best Action**.

---

## 📈 7. Scalability Strategy (Up to 100,000 Users)

To scale from this functional MVP to **100,000 active users** (~5,000–10,000 concurrent voice sessions):

1. **Stateless Horizontal Scaling:** Deploy FastAPI backend containers across **Google Cloud Run** or **AWS ECS/EKS** with CPU/concurrency-based autoscaling.
2. **Session Persistence:** Move in-memory history to a **Redis Cluster** with TTL-based expiration, allowing clients to reconnect across server instances seamlessly.
3. **Distributed Vector Database:** Transition `NumpyVectorStore` to managed **Qdrant** or **PostgreSQL + pgvector**, partitioned by `organization_id` with semantic caching.
4. **Streaming Voice Pipeline:** Transition REST audio payloads to **WebSockets / WebRTC (LiveKit)** with streaming Deepgram STT and streaming ElevenLabs TTS for sub-500ms voice round trips.
5. **Asynchronous Event Processing:** Route CRM synchronizations and call summaries through **Cloud Pub/Sub / Kafka** workers so user-facing response times remain unaffected.

---

## 🔐 8. Security, Infrastructure & Operational Economics

- **Data Privacy & Secrets:** All credentials (`.env`, `crm.db`) are excluded in `.gitignore`. Production utilizes **GCP Secret Manager**.
- **Data Encryption:** TLS 1.3 enforced for in-flight voice streams; AES-256 for database storage.
- **Unit Economics (10,000 Voice Minutes):**
  - Gemini LLM tokens: ~\$5 – \$15
  - Deepgram STT (~0.0043/min): ~\$43
  - ElevenLabs TTS (~0.015/min): ~\$150
  - Cloud Compute & Database: ~\$50
  - **Total COGS:** **~\$0.025 / minute**
  - **Commercial Feasibility:** At \$79/month for 300 minutes (Growth Plan = \$0.26/min), gross margin exceeds **88%**.
