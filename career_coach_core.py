"""Backend logic for Career Coach App.

This module is intentionally framework-independent so both the Streamlit app
and the developer notebook can import the same production backend logic.
API keys are read only from environment variables.
"""
from __future__ import annotations

import os
import re
import json
import math
import time
import textwrap
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None

SUPPORTED_DOMAINS = [
    "SDE / Software Engineering",
    "ML / AI Engineer",
    "Analyst / Consulting",
    "Core Electronics Engineer",
    "HR / Managerial Roles",
]

DOMAIN_ALIASES = {
    "sde": "SDE / Software Engineering",
    "software": "SDE / Software Engineering",
    "software engineer": "SDE / Software Engineering",
    "developer": "SDE / Software Engineering",
    "backend": "SDE / Software Engineering",
    "frontend": "SDE / Software Engineering",
    "full stack": "SDE / Software Engineering",
    "ml": "ML / AI Engineer",
    "ai": "ML / AI Engineer",
    "machine learning": "ML / AI Engineer",
    "data science": "ML / AI Engineer",
    "analyst": "Analyst / Consulting",
    "consulting": "Analyst / Consulting",
    "business analyst": "Analyst / Consulting",
    "electronics": "Core Electronics Engineer",
    "vlsi": "Core Electronics Engineer",
    "embedded": "Core Electronics Engineer",
    "communication": "Core Electronics Engineer",
    "hr": "HR / Managerial Roles",
    "manager": "HR / Managerial Roles",
    "management": "HR / Managerial Roles",
}

ROLE_RUBRICS: Dict[str, Dict[str, Any]] = {
    "SDE / Software Engineering": {
        "skills": ["python", "java", "c++", "javascript", "typescript", "react", "node", "sql", "git", "github", "oop", "dbms", "operating system", "computer networks", "api", "rest", "testing", "deployment", "docker"],
        "topics": ["data structures", "algorithms", "object oriented programming", "dbms", "operating systems", "computer networks", "system design", "apis", "testing", "deployment"],
        "project_evidence": ["web app", "backend", "frontend", "full stack", "api", "database", "authentication", "deployment", "github", "scalable", "testing"],
        "preferred_sections": ["technical skills", "projects", "experience", "achievements"],
    },
    "ML / AI Engineer": {
        "skills": ["python", "numpy", "pandas", "scikit", "tensorflow", "pytorch", "machine learning", "deep learning", "nlp", "computer vision", "mlops", "docker", "fastapi", "model evaluation", "feature engineering", "statistics"],
        "topics": ["supervised learning", "unsupervised learning", "model evaluation", "bias variance", "regularization", "feature engineering", "deep learning", "deployment", "mlops", "experiments"],
        "project_evidence": ["model", "dataset", "accuracy", "precision", "recall", "f1", "auc", "training", "inference", "deployment", "pipeline"],
        "preferred_sections": ["technical skills", "projects", "experience", "certifications"],
    },
    "Analyst / Consulting": {
        "skills": ["excel", "sql", "power bi", "tableau", "dashboard", "business analysis", "market research", "case study", "stakeholder", "consulting", "strategy", "financial", "operations", "kpi", "presentation"],
        "topics": ["structured problem solving", "business analysis", "excel", "sql analytics", "dashboarding", "market research", "case interviews", "stakeholder communication", "business impact", "presentation"],
        "project_evidence": ["dashboard", "business", "revenue", "cost", "kpi", "insight", "recommendation", "stakeholder", "market", "case", "analysis"],
        "preferred_sections": ["experience", "projects", "positions of responsibility", "achievements"],
    },
    "Core Electronics Engineer": {
        "skills": ["analog", "digital electronics", "vlsi", "verilog", "systemverilog", "embedded", "microcontroller", "arduino", "pcb", "matlab", "simulink", "signals", "communication systems", "circuit", "spice", "cadence"],
        "topics": ["network theory", "analog circuits", "digital circuits", "signals and systems", "communication systems", "microcontrollers", "embedded systems", "vlsi", "pcb design", "testing"],
        "project_evidence": ["circuit", "sensor", "microcontroller", "embedded", "pcb", "simulation", "verilog", "matlab", "signal", "communication", "hardware"],
        "preferred_sections": ["technical skills", "projects", "labs", "certifications"],
    },
    "HR / Managerial Roles": {
        "skills": ["leadership", "communication", "teamwork", "stakeholder", "planning", "recruitment", "training", "operations", "conflict", "people management", "coordination", "presentation", "negotiation"],
        "topics": ["leadership", "people management", "communication", "conflict resolution", "planning", "stakeholder management", "recruitment", "team building", "operations", "decision making"],
        "project_evidence": ["led", "managed", "coordinated", "team", "organized", "event", "stakeholder", "mentored", "planned", "resolved"],
        "preferred_sections": ["experience", "positions of responsibility", "achievements", "education"],
    },
}

ROLE_DISTANCE = {
    ("SDE / Software Engineering", "ML / AI Engineer"): 0.75,
    ("ML / AI Engineer", "SDE / Software Engineering"): 0.75,
    ("SDE / Software Engineering", "Core Electronics Engineer"): 0.45,
    ("Core Electronics Engineer", "SDE / Software Engineering"): 0.45,
    ("ML / AI Engineer", "Analyst / Consulting"): 0.55,
    ("Analyst / Consulting", "ML / AI Engineer"): 0.55,
}

SECTION_HEADERS = [
    "summary", "objective", "education", "academic profile", "skills", "technical skills", "projects", "experience", "work experience", "internship", "certifications", "achievements", "positions of responsibility", "leadership", "publications", "contact information"
]

@dataclass
class RouteDecision:
    primary_domain: str
    secondary_domain: Optional[str]
    confidence: float
    reasoning_summary: str
    attempts: int = 1
    validator_notes: List[str] = field(default_factory=list)

@dataclass
class ScoreBreakdown:
    final_score: int
    components: Dict[str, int]
    detected_direction: str
    selected_domain: str
    strong_evidence: List[str]
    missing_evidence: List[str]
    mismatch_note: str

@dataclass
class LinkedinExtraction:
    confidence: str
    content: str
    source: str
    message: str
    sections: Dict[str, str] = field(default_factory=dict)

# ----------------------------- Utilities ---------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]*", (text or "").lower())

def contains_phrase(text: str, phrase: str) -> bool:
    text_l = (text or "").lower()
    phrase_l = phrase.lower()
    if phrase_l in text_l:
        return True
    compact = re.sub(r"[^a-z0-9]+", "", text_l)
    return re.sub(r"[^a-z0-9]+", "", phrase_l) in compact

def match_terms(text: str, terms: List[str]) -> List[str]:
    return sorted({term for term in terms if contains_phrase(text, term)})

def missing_terms(text: str, terms: List[str]) -> List[str]:
    matched = set(match_terms(text, terms))
    return [t for t in terms if t not in matched]

def clamp_int(x: float, lo=0, hi=100) -> int:
    return int(max(lo, min(hi, round(x))))

def env_status() -> Dict[str, bool]:
    return {"GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")), "SERPAPI_KEY": bool(os.getenv("SERPAPI_KEY"))}

# ----------------------------- Resume parsing ----------------------------

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from a Streamlit UploadedFile or file path."""
    name = getattr(uploaded_file, "name", None) or str(uploaded_file)
    ext = name.lower().split(".")[-1]
    data = None
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        with open(uploaded_file, "rb") as f:
            data = f.read()
    if ext == "pdf":
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except Exception:
            try:
                from pypdf import PdfReader
                import io
                reader = PdfReader(io.BytesIO(data))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                raise RuntimeError(f"Could not read PDF: {e}")
    if ext == "docx":
        try:
            import docx
            import io
            d = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in d.paragraphs)
        except Exception as e:
            raise RuntimeError(f"Could not read DOCX: {e}")
    return data.decode("utf-8", errors="ignore")

def extract_resume_sections(resume_text: str) -> Dict[str, str]:
    lines = [ln.strip() for ln in (resume_text or "").splitlines() if ln.strip()]
    sections: Dict[str, List[str]] = {"profile": []}
    current = "profile"
    header_pattern = re.compile(r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:?$", re.I)
    for ln in lines:
        clean = re.sub(r"[^a-zA-Z ]", "", ln).strip().lower()
        m = header_pattern.match(clean)
        if m or (len(ln) < 35 and clean in SECTION_HEADERS):
            current = clean
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(ln)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}

def summarize_resume(resume_text: str, max_chars: int = 3500) -> str:
    text = normalize_text(resume_text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."

# ----------------------------- Domain intelligence -----------------------

def normalize_domain(selection: str) -> str:
    s = (selection or "").lower()
    for key, domain in DOMAIN_ALIASES.items():
        if key in s:
            return domain
    for d in SUPPORTED_DOMAINS:
        if d.lower() == s:
            return d
    return selection if selection in SUPPORTED_DOMAINS else "SDE / Software Engineering"

def domain_evidence_scores(resume_text: str) -> Dict[str, int]:
    text = resume_text or ""
    scores = {}
    for domain, rubric in ROLE_RUBRICS.items():
        skills = match_terms(text, rubric["skills"])
        topics = match_terms(text, rubric["topics"])
        projects = match_terms(text, rubric["project_evidence"])
        score = len(skills) * 3 + len(topics) * 2 + len(projects) * 2
        scores[domain] = score
    return scores

def detect_resume_direction(resume_text: str) -> Tuple[str, Dict[str, int]]:
    scores = domain_evidence_scores(resume_text)
    if not scores or max(scores.values()) == 0:
        return "General / Undetermined", scores
    return max(scores, key=scores.get), scores

def orchestrate_domain(resume_text: str, selected_role: str, requested_feature: str = "general") -> RouteDecision:
    selected_domain = normalize_domain(selected_role)
    detected, scores = detect_resume_direction(resume_text)
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    secondary = sorted_domains[0][0] if sorted_domains else None
    if secondary == selected_domain and len(sorted_domains) > 1:
        secondary = sorted_domains[1][0]

    # User's target role wins, but resume evidence influences confidence and notes.
    selected_score = scores.get(selected_domain, 0)
    top_score = max(scores.values()) if scores else 0
    confidence = 0.75 if selected_score >= max(3, top_score * 0.55) else 0.62
    if requested_feature.lower() in {"resume", "ats", "roadmap", "linkedin", "interview"}:
        confidence += 0.05
    confidence = min(confidence, 0.94)
    decision = RouteDecision(
        primary_domain=selected_domain,
        secondary_domain=secondary,
        confidence=confidence,
        reasoning_summary=f"Routed to {selected_domain} because the selected target role is the primary constraint. Resume direction appears to be {detected}.",
    )
    return validate_and_repair_route(decision, resume_text, selected_domain, requested_feature)

def validate_route(decision: RouteDecision, resume_text: str, selected_domain: str, requested_feature: str) -> Tuple[bool, str]:
    if decision.primary_domain not in SUPPORTED_DOMAINS:
        return False, "Primary domain is outside supported domains."
    if decision.primary_domain != selected_domain:
        return False, "Selected user target role was not respected."
    if not (0 <= decision.confidence <= 1):
        return False, "Confidence score is invalid."
    if requested_feature == "interview" and not decision.primary_domain:
        return False, "Interview routing needs a role domain."
    return True, "Valid route."

def validate_and_repair_route(decision: RouteDecision, resume_text: str, selected_domain: str, requested_feature: str, max_attempts: int = 3) -> RouteDecision:
    notes = []
    current = decision
    for attempt in range(1, max_attempts + 1):
        valid, note = validate_route(current, resume_text, selected_domain, requested_feature)
        notes.append(f"Attempt {attempt}: {note}")
        current.attempts = attempt
        if valid:
            current.validator_notes = notes
            return current
        current.primary_domain = selected_domain
        current.confidence = min(max(current.confidence, 0.55), 0.9)
    current.validator_notes = notes
    current.primary_domain = selected_domain
    current.reasoning_summary += " Validator fallback used selected target role as final route."
    return current

# ----------------------------- LLM ---------------------------------------

class LLMClient:
    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        # Used only for coding / mathematical reasoning responses. Kept backend-only.
        self.code_model = os.getenv("GROQ_CODE_MODEL", "qwen/qwen3-32b")
        self.client = None
        if self.api_key and Groq is not None:
            self.client = Groq(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self.client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.25,
        max_tokens: int = 1200,
        model: Optional[str] = None,
        task_type: str = "general",
    ) -> str:
        if not self.available:
            return self._fallback(messages)
        selected_model = model or (self.code_model if task_type in {"coding", "math", "coding_solution"} else self.model)
        try:
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            # If the code-specialized model is unavailable in a user's Groq account, fall back to the default model.
            if task_type in {"coding", "math", "coding_solution"} and selected_model != self.model:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content or ""
                except Exception as fallback_error:
                    return f"The AI service could not complete the request right now. Backend error: {fallback_error}"
            return f"The AI service could not complete the request right now. Backend error: {e}"

    def _fallback(self, messages: List[Dict[str, str]]) -> str:
        return (
            "AI generation is unavailable because GROQ_API_KEY is not configured on the server. "
            "The app can still show rule-based diagnostics. Configure GROQ_API_KEY as an environment variable for full responses."
        )

# ----------------------------- Scoring -----------------------------------

def role_fit_breakdown(resume_text: str, selected_domain: str) -> ScoreBreakdown:
    selected_domain = normalize_domain(selected_domain)
    rubric = ROLE_RUBRICS[selected_domain]
    detected_direction, domain_scores = detect_resume_direction(resume_text)
    text = resume_text or ""

    skill_matches = match_terms(text, rubric["skills"])
    topic_matches = match_terms(text, rubric["topics"])
    project_matches = match_terms(text, rubric["project_evidence"])
    missing = missing_terms(text, rubric["skills"][:14])

    role_alignment = clamp_int((len(skill_matches) / max(1, len(rubric["skills"]))) * 30, 0, 30)
    skill_score = clamp_int((len(skill_matches) / max(1, min(len(rubric["skills"]), 12))) * 20, 0, 20)
    project_score = clamp_int((len(project_matches) / max(1, min(len(rubric["project_evidence"]), 8))) * 15, 0, 15)

    # Impact signals: numbers, metrics, action words.
    metric_count = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|x|k|users|samples|projects|members|hours|ms|seconds|accuracy|f1|revenue|cost)?\b", text.lower()))
    action_words = ["built", "developed", "led", "improved", "optimized", "deployed", "designed", "analyzed", "implemented", "created", "managed"]
    action_count = len(match_terms(text, action_words))
    impact_score = clamp_int(min(15, metric_count * 2 + action_count), 0, 15)

    sections = extract_resume_sections(resume_text)
    structure_score = 10 if len(sections) >= 4 else 7 if len(sections) >= 2 else 4
    ats_score = clamp_int((len(topic_matches) / max(1, min(len(rubric["topics"]), 8))) * 10, 0, 10)

    # Mismatch penalty when resume direction differs substantially from selected domain.
    mismatch_note = ""
    raw_total = role_alignment + skill_score + project_score + impact_score + structure_score + ats_score
    if detected_direction != "General / Undetermined" and detected_direction != selected_domain:
        similarity = ROLE_DISTANCE.get((detected_direction, selected_domain), 0.35)
        # If selected evidence is weak, reduce inflated generic scores.
        if len(skill_matches) < 5 and len(project_matches) < 3:
            penalty = int((1 - similarity) * 18)
            raw_total = max(0, raw_total - penalty)
            mismatch_note = f"Resume direction appears closer to {detected_direction}; selected goal is {selected_domain}, so a mismatch penalty was applied."
        else:
            mismatch_note = f"Resume direction appears closer to {detected_direction}, but there is enough selected-role evidence to avoid a heavy penalty."

    strong = skill_matches[:8] + project_matches[:5]
    if not strong:
        strong = ["Basic resume structure detected"] if structure_score >= 7 else []
    return ScoreBreakdown(
        final_score=clamp_int(raw_total),
        components={
            "Role alignment": role_alignment,
            "Required skills match": skill_score,
            "Project/experience relevance": project_score,
            "Impact and metrics": impact_score,
            "Structure/readability": structure_score,
            "ATS keywords": ats_score,
        },
        detected_direction=detected_direction,
        selected_domain=selected_domain,
        strong_evidence=strong[:12],
        missing_evidence=missing[:12],
        mismatch_note=mismatch_note,
    )

def generate_resume_report(resume_text: str, selected_domain: str, llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    selected_domain = normalize_domain(selected_domain)
    breakdown = role_fit_breakdown(resume_text, selected_domain)
    llm = llm or LLMClient()
    system = "You are a strict role-specific resume evaluator. Do not inflate scores. Explain evidence-backed fixes only."
    user = f"""
Selected role/domain: {selected_domain}
Detected resume direction: {breakdown.detected_direction}
Final backend score: {breakdown.final_score}/100
Component scores: {json.dumps(breakdown.components)}
Strong evidence: {breakdown.strong_evidence}
Missing evidence: {breakdown.missing_evidence}
Mismatch note: {breakdown.mismatch_note}
Resume excerpt:\n{summarize_resume(resume_text)}

Write a concise professional resume improvement report with:
1. Fit assessment
2. Top priority fixes
3. Section-wise issues
4. 3 bullet rewrite examples tailored to {selected_domain}
Do not change the backend score.
"""
    analysis = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=1400)
    return {"breakdown": breakdown, "analysis": analysis}

def ats_analysis(resume_text: str, selected_domain: str, job_description: str = "", llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    selected_domain = normalize_domain(selected_domain)
    base = role_fit_breakdown(resume_text + "\n" + job_description, selected_domain)
    rubric = ROLE_RUBRICS[selected_domain]
    jd_terms = tokenize(job_description)
    jd_keywords = sorted(set([w for w in jd_terms if len(w) > 3]))[:35]
    role_terms = rubric["skills"] + rubric["topics"] + jd_keywords
    matched = match_terms(resume_text, role_terms)
    missing = [t for t in role_terms if t not in matched][:18]
    keyword_score = clamp_int((len(matched) / max(1, min(len(role_terms), 35))) * 25, 0, 25)
    final = clamp_int(base.final_score * 0.75 + keyword_score)
    llm = llm or LLMClient()
    prompt = f"""
You are an ATS evaluator. Use the backend score as final score, do not invent a different score.
Selected domain: {selected_domain}
Final ATS score: {final}/100
Matched terms: {matched[:20]}
Missing terms: {missing[:18]}
Component score: {base.components}
Resume excerpt: {summarize_resume(resume_text)}
Optional job description: {job_description[:2000]}

Return:
- one-line fit summary
- strong matches
- missing/weak keywords
- priority fixes
- rewritten keyword-rich summary line
"""
    report = llm.chat([{"role":"system","content":"You are a strict ATS and role-fit checker."},{"role":"user","content":prompt}], max_tokens=1200)
    return {"score": final, "matched": matched[:20], "missing": missing[:18], "report": report, "breakdown": base}

# ----------------------------- SERP resources -----------------------------

def serp_search(query: str, num: int = 5) -> List[Dict[str, str]]:
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key or requests is None:
        return []
    try:
        params = {"engine": "google", "q": query, "api_key": api_key, "num": num}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=12)
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        for item in data.get("organic_results", [])[:num]:
            results.append({
                "title": item.get("title", "Untitled"),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return results
    except Exception:
        return []

def resource_queries(domain: str, level: str, preference: str) -> List[str]:
    domain = normalize_domain(domain)
    topics = ROLE_RUBRICS[domain]["topics"][:5]
    if level == "Beginner":
        suffix = "beginner tutorial lecture playlist"
    elif level == "Intermediate":
        suffix = "practice questions projects tutorial"
    else:
        suffix = "advanced projects interview questions system design case study"
    pref = preference.lower()
    if "video" in pref:
        suffix += " youtube"
    elif "documentation" in pref:
        suffix += " official documentation guide"
    elif "practice" in pref:
        suffix += " problems practice"
    elif "project" in pref:
        suffix += " project github"
    return [f"{domain} {topic} {level} {suffix}" for topic in topics[:4]]

def fetch_ranked_resources(domain: str, level: str, preference: str, max_results: int = 8) -> List[Dict[str, str]]:
    all_results = []
    seen = set()
    for q in resource_queries(domain, level, preference):
        for res in serp_search(q, num=3):
            link = res.get("link", "")
            if link and link not in seen:
                seen.add(link)
                res["query"] = q
                all_results.append(res)
            if len(all_results) >= max_results:
                break
        if len(all_results) >= max_results:
            break
    return all_results

CURATED_RESOURCES = {
    "SDE / Software Engineering": {
        "Beginner": ["CS50 or equivalent programming foundations", "NeetCode beginner arrays/strings", "Git and GitHub official getting started"],
        "Intermediate": ["LeetCode patterns", "DBMS + OS + CN interview notes", "Build REST API with database and auth"],
        "Advanced": ["System design primer", "Advanced graph/DP practice", "Deploy tested full-stack project"],
    },
    "ML / AI Engineer": {
        "Beginner": ["Python, NumPy, Pandas basics", "Andrew Ng Machine Learning foundations", "Kaggle beginner notebooks"],
        "Intermediate": ["Hands-on ML projects", "Model evaluation and feature engineering", "ML interview questions by topic"],
        "Advanced": ["MLOps deployment pipeline", "Deep learning specialization topics", "Model monitoring and serving case studies"],
    },
    "Analyst / Consulting": {
        "Beginner": ["Excel fundamentals", "SQL basics", "Business problem framing basics"],
        "Intermediate": ["Dashboard project", "Case interview practice", "SQL analytics problems"],
        "Advanced": ["Consulting case books", "Market sizing drills", "Executive presentation storytelling"],
    },
    "Core Electronics Engineer": {
        "Beginner": ["Network theory basics", "Analog and digital electronics lectures", "MATLAB basics"],
        "Intermediate": ["Microcontroller projects", "Verilog/VLSI practice", "Communication systems problems"],
        "Advanced": ["PCB design project", "Advanced VLSI verification", "Embedded system deployment"],
    },
    "HR / Managerial Roles": {
        "Beginner": ["Communication fundamentals", "HR basics", "Leadership behavior examples"],
        "Intermediate": ["Recruitment case practice", "People analytics basics", "Conflict resolution scenarios"],
        "Advanced": ["Org design cases", "Stakeholder management simulations", "Leadership portfolio building"],
    },
}

def generate_roadmap(resume_text: str, selected_domain: str, level: str, duration_weeks: int, weekly_hours: int, preference: str, goal: str, llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    selected_domain = normalize_domain(selected_domain)
    level = level if level in ["Beginner", "Intermediate", "Advanced"] else "Beginner"
    breakdown = role_fit_breakdown(resume_text, selected_domain)
    live_resources = fetch_ranked_resources(selected_domain, level, preference)
    fallback_resources = CURATED_RESOURCES[selected_domain][level]
    resources_text = json.dumps(live_resources[:8], indent=2) if live_resources else "\n".join(f"- {x}" for x in fallback_resources)
    balance = {
        "Beginner": "70% concepts, 20% guided practice, 10% mini-projects",
        "Intermediate": "35% concept strengthening, 40% practice, 25% projects",
        "Advanced": "20% revision, 30% advanced practice, 40% projects/system design, 10% portfolio polish",
    }[level]
    llm = llm or LLMClient()
    prompt = f"""
Create a role-specific roadmap. The selected current level is the primary constraint.
Do not infer a higher/lower level only from the resume.
Domain: {selected_domain}
Current level: {level}
Duration: {duration_weeks} weeks
Weekly hours: {weekly_hours}
Preference: {preference}
Goal: {goal}
Planning balance: {balance}
Detected resume direction: {breakdown.detected_direction}
Strong evidence: {breakdown.strong_evidence}
Missing evidence: {breakdown.missing_evidence}
Resources available:\n{resources_text}

Output:
1. Readiness diagnosis
2. Skill gap summary
3. Week-by-week roadmap with learn/practice/build/checkpoint
4. Resources per phase
5. Final portfolio/interview deliverables
Make Beginner, Intermediate, and Advanced roadmaps clearly different.
"""
    roadmap = llm.chat([{"role":"system","content":"You are a senior career roadmap architect. Produce specific, measurable plans."},{"role":"user","content":prompt}], max_tokens=1800)
    return {"roadmap": roadmap, "resources": live_resources, "used_live_resources": bool(live_resources), "breakdown": breakdown}

# ----------------------------- LinkedIn ----------------------------------

def validate_linkedin_url(url: str) -> bool:
    try:
        p = urlparse(url if url.startswith("http") else "https://" + url)
        return "linkedin.com" in p.netloc.lower() and "/in/" in p.path.lower()
    except Exception:
        return False

def fetch_linkedin_profile_content(url: str) -> LinkedinExtraction:
    if not validate_linkedin_url(url):
        return LinkedinExtraction("LOW", "", "validation", "Enter a valid public LinkedIn profile URL containing linkedin.com/in/.")
    # Direct public fetch. LinkedIn often blocks this; handle honestly.
    content_parts = []
    if requests is not None and BeautifulSoup is not None:
        try:
            headers = {"User-Agent": "Mozilla/5.0 CareerCoachBot/1.0"}
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200 and len(r.text) > 500:
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.extract()
                text = normalize_text(soup.get_text(" "))
                if len(text) > 800 and any(x in text.lower() for x in ["experience", "education", "skills", "linkedin"]):
                    return LinkedinExtraction("MEDIUM", text[:6000], "direct public page", "Public LinkedIn content was partially extracted.")
        except Exception:
            pass
    # SerpAPI fallback: public snippets only.
    results = serp_search(f"site:linkedin.com/in {url}", num=5)
    if results:
        snippets = "\n".join(f"{r.get('title','')}\n{r.get('snippet','')}\n{r.get('link','')}" for r in results)
        if len(snippets.strip()) > 80:
            return LinkedinExtraction("MEDIUM", snippets, "search snippets", "Only public/search-visible LinkedIn information could be analyzed.")
    return LinkedinExtraction("LOW", "", "none", "Automatic LinkedIn extraction was limited. Paste only the sections you want optimized.")

def linkedin_manual_sections(**sections: str) -> Dict[str, str]:
    return {k: v.strip() for k, v in sections.items() if v and v.strip()}

def generate_linkedin_optimization(resume_text: str, selected_domain: str, linkedin_url: str, extraction: LinkedinExtraction, manual_sections: Optional[Dict[str, str]] = None, llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    selected_domain = normalize_domain(selected_domain)
    manual_sections = manual_sections or {}
    content = extraction.content or "\n".join(f"{k}: {v}" for k, v in manual_sections.items())
    llm = llm or LLMClient()
    if not content.strip():
        return {"status": "needs_manual_sections", "message": extraction.message, "report": ""}
    prompt = f"""
Optimize a LinkedIn profile for {selected_domain}.
URL: {linkedin_url}
Extraction confidence: {extraction.confidence}
Extraction source: {extraction.source}
Resume excerpt: {summarize_resume(resume_text)}
LinkedIn/manual profile content:\n{content[:6000]}

Return:
- LinkedIn profile score out of 100
- Search visibility issues
- Resume/profile consistency issues
- 5 headline options
- Rewritten About section
- Experience/project bullet improvements
- Skills to add/reorder
- Featured section recommendations
Be honest if only snippets/manual sections were analyzed.
"""
    report = llm.chat([{"role":"system","content":"You are a LinkedIn optimization specialist. Never claim full profile access unless content is available."},{"role":"user","content":prompt}], max_tokens=1600)
    return {"status": "ok", "message": extraction.message, "report": report, "confidence": extraction.confidence}

# ----------------------------- Interview engine --------------------------

QUESTION_TOPICS = {
    "DSA": {
        "Beginner": ["arrays", "strings", "hash maps", "time complexity"],
        "Intermediate": ["two pointers", "sliding window", "trees", "graphs", "recursion"],
        "Advanced": ["dynamic programming", "advanced graphs", "tries", "greedy proofs", "systematic optimization"],
    },
    "ML": {
        "Beginner": ["supervised vs unsupervised learning", "train/test split", "overfitting", "basic metrics"],
        "Intermediate": ["feature engineering", "regularization", "class imbalance", "model selection", "precision recall"],
        "Advanced": ["model monitoring", "MLOps", "deep learning optimization", "data leakage", "production inference"],
    },
    "HR": {
        "Beginner": ["introduce yourself", "strengths", "teamwork", "motivation"],
        "Intermediate": ["conflict", "failure", "deadlines", "leadership", "feedback"],
        "Advanced": ["ambiguous situations", "stakeholder conflict", "ethical decisions", "influence without authority"],
    },
    "Resume Deep Dive": {
        "Beginner": ["project overview", "your contribution", "tools used"],
        "Intermediate": ["tradeoffs", "metrics", "challenges", "design decisions"],
        "Advanced": ["scalability", "failure modes", "architecture", "future improvements"],
    },
}

DIFF_ORDER = ["Beginner", "Intermediate", "Advanced"]

def next_difficulty(current: str, score: int) -> str:
    idx = DIFF_ORDER.index(current) if current in DIFF_ORDER else 0
    if score >= 8:
        idx = min(2, idx + 1)
    elif score < 5:
        idx = max(0, idx - 1)
    return DIFF_ORDER[idx]

def generate_interview_question(mode: str, selected_domain: str, difficulty: str, resume_text: str, previous_questions: List[str], llm: Optional[LLMClient] = None, coding_language: str = "Python") -> str:
    mode = mode if mode in QUESTION_TOPICS else "DSA"
    selected_domain = normalize_domain(selected_domain)
    difficulty = difficulty if difficulty in DIFF_ORDER else "Beginner"
    topics = QUESTION_TOPICS[mode][difficulty]
    resume_rule = "Use resume heavily and ask project-specific questions." if mode == "Resume Deep Dive" else "Do not use resume details except light personalization. Ask general important questions for the selected mode."
    resume_context = summarize_resume(resume_text, 2500) if mode == "Resume Deep Dive" else "Resume available but should not dominate this mode."
    llm = llm or LLMClient()
    prompt = f"""
Generate exactly one interview question.
Mode: {mode}
Target role/domain: {selected_domain}
Difficulty: {difficulty}
Allowed topics: {topics}
Resume rule: {resume_rule}
Previous questions to avoid: {previous_questions[-8:]}
Resume context: {resume_context}

Rules:
- Ask one question only.
- Do not repeat previous questions.
- For DSA, ask a coding/problem-solving question and mention that the candidate may answer in {coding_language}.
- For ML, ask a conceptual/applied ML question.
- For HR, ask a behavioral STAR-style question.
- For Resume Deep Dive, ask from resume and target role.
"""
    return llm.chat([{"role":"system","content":"You are an adaptive interview coach."},{"role":"user","content":prompt}], max_tokens=350).strip()


def enforce_coding_answer_format(ideal_answer: str, language_name: str) -> str:
    """Lightweight guardrail so DSA ideal answers stay readable and language-tagged."""
    fence_map = {"Python": "python", "C++": "cpp", "Java": "java"}
    fence = fence_map.get(language_name, "python")
    text = (ideal_answer or "").strip()
    if not text:
        return text
    # If the model forgot a code fence but appears to include code, wrap the code-like part.
    if "```" not in text and ("Code:" in text or "class " in text or "def " in text or "public " in text or "#include" in text):
        if "Code:" in text:
            before, after = text.split("Code:", 1)
            return f"{before.strip()}\n\nCode:\n```{fence}\n{after.strip()}\n```"
        return f"Code:\n```{fence}\n{text}\n```"
    # Normalize common generic fences to the selected language fence.
    text = re.sub(r"```(?:python|py|cpp|c\+\+|java|javascript|js)?", f"```{fence}", text, count=1, flags=re.I)
    return text

def evaluate_interview_answer(mode: str, selected_domain: str, question: str, answer: str, difficulty: str, llm: Optional[LLMClient] = None, coding_language: str = "Python") -> Dict[str, Any]:
    llm = llm or LLMClient()
    is_coding = mode == "DSA"
    language_name = coding_language if coding_language in {"Python", "C++", "Java"} else "Python"
    language_fence = {"Python": "python", "C++": "cpp", "Java": "java"}[language_name]

    if is_coding:
        ideal_answer_contract = f"""
For ideal_answer, return markdown with this exact structure and readable formatting.
The code MUST be written only in {language_name}. Do not switch to another language.
Use the exact markdown fence ```{language_fence}. Do not compress code into one line.

Approach:
- Explain the algorithm in 3-6 bullet points.

Code:
```{language_fence}
<clean, complete, properly indented {language_name} solution only>
```

Time Complexity:
- O(...)

Space Complexity:
- O(...)

Edge Cases:
- Mention important edge cases.
"""
    else:
        ideal_answer_contract = "For ideal_answer, provide a concise structured answer with bullet points. Do not include code unless the question specifically requires it."

    prompt = f"""
Evaluate the candidate answer.
Mode: {mode}
Domain: {selected_domain}
Difficulty: {difficulty}
Question: {question}
Candidate answer: {answer}
Selected coding language: {language_name}

Important coding-language rule:
- If Mode is DSA, the ideal answer code must be in {language_name} only.
- Do not output Python when {language_name} is C++ or Java.
- Do not output C++ when {language_name} is Python or Java.
- The markdown code fence must be ```{language_fence}.

Scoring rules:
- Score from 0 to 10.
- For DSA/coding, reward correct algorithm, correctness, complexity, edge cases, and code quality.
- Do not reduce adaptive difficulty logic. The next difficulty is handled by backend.

{ideal_answer_contract}

Return JSON only with keys:
score: integer 0-10
feedback: concise feedback
missing_points: list of missing points
ideal_answer: markdown string
"""
    raw = llm.chat(
        [{"role":"system","content":"You are a strict interview evaluator. Return valid JSON only. For coding ideal answers, preserve markdown code fences inside the JSON string."},{"role":"user","content":prompt}],
        temperature=0.1,
        max_tokens=1400 if is_coding else 900,
        task_type="coding_solution" if is_coding else "general",
    )
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("No JSON object found")
        data = json.loads(match.group(0))
        score = int(data.get("score", 5))
        data["score"] = max(0, min(10, score))
        data["feedback"] = str(data.get("feedback", ""))
        mp = data.get("missing_points", [])
        data["missing_points"] = mp if isinstance(mp, list) else [str(mp)]
        data["ideal_answer"] = enforce_coding_answer_format(str(data.get("ideal_answer", "")), language_name) if is_coding else str(data.get("ideal_answer", ""))
    except Exception:
        # Rule-based fallback if LLM/JSON fails. Keep adaptive difficulty working.
        length = len(tokenize(answer))
        score = 8 if length > 80 else 6 if length > 35 else 4
        fallback_ideal = "Configure GROQ_API_KEY for detailed ideal answers."
        if is_coding:
            fallback_ideal = f"""Approach:
- Explain the algorithm clearly.
- Provide a complete {language_name} solution.
- Discuss complexity and edge cases.

Code:
```{language_fence}
// Ideal solution unavailable because the AI response could not be parsed.
```

Time Complexity:
- Depends on the chosen approach.

Space Complexity:
- Depends on the chosen approach.

Edge Cases:
- Empty input, single element, duplicates, negative values, and boundary constraints where applicable.
"""
        data = {"score": score, "feedback": raw if raw else "Answer received.", "missing_points": [], "ideal_answer": fallback_ideal}
    data["next_difficulty"] = next_difficulty(difficulty, data["score"])
    return data
