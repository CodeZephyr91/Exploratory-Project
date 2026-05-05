# Career Coach App

A mature Streamlit career preparation app with role-specific agents for:

- SDE / Software Engineering
- ML / AI Engineer
- Analyst / Consulting
- Core Electronics Engineer
- HR / Managerial Roles

## Security

No API key is displayed or requested in the frontend. Keys are read only from server environment variables:

```bash
GROQ_API_KEY
GROQ_MODEL
GROQ_CODE_MODEL
SERPAPI_KEY
```

`SERPAPI_KEY` is optional. If missing, roadmap and LinkedIn use safe fallback behavior.

## Run

```bash
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_key"
$env:GROQ_MODEL="llama-3.3-70b-versatile"
$env:GROQ_CODE_MODEL="qwen/qwen3-32b"
$env:SERPAPI_KEY="your_serpapi_key"
streamlit run app.py
```

Or copy `.env.example` to `.env` and fill values.

## Architecture

`career_coach_core.py` contains the shared backend logic used by both the Streamlit website and the developer notebook.

The app includes:

- Multi-agent domain orchestrator
- Validator with max 3 route correction attempts
- Role-specific resume scoring
- Hybrid ATS scoring
- Level-aware roadmap generator with live resource search where available
- LinkedIn URL extraction with fallback manual section optimization
- Adaptive interview prep for DSA, ML, HR, and Resume Deep Dive
- DSA coding mode with Python, C++, and Java answer-language selection
- Readable ideal answer rendering with syntax-highlighted code blocks
- Optional Qwen 32B routing for coding/math-heavy answer generation via `GROQ_CODE_MODEL`
