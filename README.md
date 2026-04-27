# F1 Race Intelligence Platform 🏎️💨

A high-performance race visualization and intelligence platform that combines **FastF1 telemetry**, **LLM-powered RAG insights**, and **3D track animation**.

## 🌟 Key Features

- **3D Track Visualization**: Interactive race track with real-time car positioning and lap-by-lap playback.
- **AI Pit Crew**: A RAG-powered chatbot that analyzes driver performance, race strategy, and historical data.
- **Telemetry Analytics**: Real-time charts for speed, throttle, brake, and gear usage.
- **Dynamic Insights**: Automatically generated race highlights and strategy summaries.
- **Video Integration**: Synchronized YouTube race highlights with telemetry playback.

---

## 🏗️ System Architecture

The project follows a modern decoupled architecture with a FastAPI backend and a React frontend.

```mermaid
graph TD
    User([User]) <--> Frontend[React/D3 Frontend]
    Frontend <--> Backend[FastAPI Backend]
    
    subgraph "Intelligent Logic"
        Backend <--> Simulation[Tier-2 Prediction Engine]
        Backend <--> RAG[Local RAG Service]
        RAG <--> VectorDB[(SQLite/Persistent Cache)]
        Backend <--> Ollama[Local Ollama (Llama-3.2)]
        Backend <--> Gemini[Gemini 2.0 Fallback]
        Backend <--> OR[OpenRouter Safety Net]
    end
    
    subgraph "Data Sources"
        Backend <--> FF1[FastF1 Telemetry Engine]
        Backend <--> Scraping[Wiki/Web Scrapers]
        FF1 <--> LocalCache[(Local Parquet/JSON)]
    end
```

---

## 🛠️ Data Ingestion & RAG Workflow

The system uses a sophisticated pipeline to turn raw telemetry and text into actionable insights.

```mermaid
sequenceDiagram
    participant User
    participant API as Chat API
    participant RAG as RAG Service
    participant DB as pgvector
    participant LLM as AI Model

    User->>API: "Who had the pole at Bahrain 2024?"
    API->>RAG: Query "Bahrain 2024 Pole"
    RAG->>DB: Search Embeddings
    DB-->>RAG: Return Top Context
    RAG->>LLM: Prompt + Local Stats + Context
    LLM-->>API: "Max Verstappen took pole..."
    API-->>User: [Full Driver Name Answer]
```

---

## 📁 Project Structure

### Backend (`/src/backend`)
- **`routers/`**: FastAPI endpoints (Session, Track, Chat, Ingest).
- **`services/`**: Core logic providers.
  - `fetcher.py`: Manages FastF1 data loading and caching.
  - `laps.py`: Processes raw telemetry into discrete lap summaries.
  - `local_rag.py`: Orchestrates local search and LLM context injection.
  - `video.py`: Handles YouTube synchronization.

### Frontend (`/src/frontend`)
- **`components/`**: Modular UI elements (Bento Grid, Highlight Cards).
- **`components/track/`**: 3D Track logic and car layer animations.
- **`store/`**: Zustand state management for telemetry playback.

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- **Ollama**: Installed locally on the host machine.
- API Keys: `F1VIZ_GEMINI_API_KEY` and `F1VIZ_OPENROUTER_API_KEY`.

### Quick Start (Ollama + Docker)
1.  **Start Ollama**:
    Ensure the Ollama service is running on your host machine:
    ```bash
    ollama serve
    ollama pull llama3.2:latest
    ```

2.  **Configure Environment**:
    Create a `.env` file in the root with the following provider chain settings:
    ```env
    # LLM Providers
    F1VIZ_OLLAMA_ENABLED=true
    F1VIZ_OLLAMA_MODEL=llama3.2:latest
    F1VIZ_GEMINI_API_KEY=your_gemini_key
    F1VIZ_OPENROUTER_API_KEY=your_openrouter_key
    
    # Cascade Order: Ollama -> OpenRouter -> Gemini
    ```

3.  **Launch Platform**:
    ```bash
    docker-compose up --build
    ```
    - **Frontend**: http://localhost:3000
    - **Backend API Docs**: http://localhost:8000/docs

### 🧪 Verification
To verify the physics engine and LLM routing logic:
```bash
docker-compose exec backend pytest tests/test_prediction_engine_tier2.py -v
```

### Local Development
**Backend**:
```bash
pip install -r requirements.txt
uvicorn src.backend.app:app --reload
```

**Frontend**:
```bash
cd src/frontend
npm install
npm run dev
```

---

## 📊 Core Functions & Utilities

### Driver Normalization
The system employs a global mapping utility (`_DRIVER_CODE_MAPPING`) to ensure that all internal three-letter codes (e.g., `VER`) are resolved to professional full names (e.g., `Max Verstappen`) in all AI responses and UI labels.

### Session Sync
Telemetry data is resampled to a consistent 10Hz frequency to ensure smooth 3D animations and precise synchronization with video playback timestamps.

---

## ✨ Credits
- **Data**: Powered by [FastF1](https://github.com/theOehrly/FastF1)
- **Intelligence**: Google Gemini / OpenAI
- **Visualization**: React, D3.js, Lucide Icons
