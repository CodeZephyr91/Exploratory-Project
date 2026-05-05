import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from career_coach_core import (
    SUPPORTED_DOMAINS, env_status, extract_text_from_file, extract_resume_sections,
    orchestrate_domain, ats_analysis, generate_roadmap, fetch_linkedin_profile_content,
    generate_linkedin_optimization, linkedin_manual_sections, generate_interview_question,
    evaluate_interview_answer, generate_resume_report, role_fit_breakdown, LLMClient,
)

load_dotenv()

st.set_page_config(page_title="Career Coach App", page_icon=None, layout="wide")

st.markdown("""
<style>
:root { --primary:#1f3554; --muted:#697386; --bg:#f7f9fc; --border:#dfe5ee; --accent:#2f5f98; }
.stApp { background: #ffffff; }
.main .block-container { max-width: 1180px; padding-top: 2.2rem; }
section[data-testid="stSidebar"] { background: #f3f6fa; border-right: 1px solid #e1e6ef; }
h1, h2, h3 { color: #172033; letter-spacing: -0.02em; }
.hero { background: linear-gradient(135deg, #18263d 0%, #2a4266 100%); padding: 34px 40px; border-radius: 20px; margin-bottom: 24px; box-shadow: 0 14px 35px rgba(31,53,84,.12); }
.hero h1 { color: white; margin: 0; font-size: 2.1rem; }
.hero p { color: #dce7f5; margin-top: 12px; font-size: 1rem; max-width: 860px; }
.card { border: 1px solid var(--border); border-radius: 16px; padding: 18px 20px; background: white; box-shadow: 0 8px 24px rgba(16,24,40,.04); }
.metric-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { color: #172033; font-size: 1.25rem; font-weight: 700; margin-top: 6px; }
.notice { border-left: 4px solid var(--accent); padding: 12px 16px; background:#f6f9fd; border-radius: 10px; color:#22314c; }
.small-muted { color: var(--muted); font-size:.9rem; }
div.stButton > button:first-child { background:#1f3554; color:white; border-radius:10px; padding:.65rem 1.1rem; border:0; }
div.stButton > button:first-child:hover { background:#284871; color:white; border:0; }
[data-testid="stMetricValue"] { color:#172033; }
</style>
""", unsafe_allow_html=True)

# ----------------------- Session init -----------------------
if "resume_text" not in st.session_state: st.session_state.resume_text = ""
if "resume_name" not in st.session_state: st.session_state.resume_name = ""
if "route" not in st.session_state: st.session_state.route = None
if "interview" not in st.session_state:
    st.session_state.interview = {"active": False, "mode": None, "difficulty": "Beginner", "questions": [], "answers": [], "evaluations": [], "current_question": "", "coding_language": "Python"}
if "linkedin_extraction" not in st.session_state: st.session_state.linkedin_extraction = None

# ----------------------- Sidebar ----------------------------
st.sidebar.title("Workspace")
st.sidebar.caption("Shared inputs for all modules")
resume_file = st.sidebar.file_uploader("Upload resume once", type=["pdf", "docx", "txt", "md"])
if resume_file is not None:
    try:
        text = extract_text_from_file(resume_file)
        st.session_state.resume_text = text
        st.session_state.resume_name = resume_file.name
        st.sidebar.success(f"Resume loaded: {resume_file.name}")
    except Exception as e:
        st.sidebar.error(str(e))

target_domain = st.sidebar.selectbox("Target domain", SUPPORTED_DOMAINS, index=0)
target_role = st.sidebar.text_input("Specific target role", value="Software Development Engineer")
company_domain = st.sidebar.text_input("Target company or domain", value="")
st.sidebar.divider()
st.sidebar.subheader("Backend status")
status = env_status()
if status["GROQ_API_KEY"]:
    st.sidebar.success("Groq configured")
else:
    st.sidebar.warning("Groq key missing on server")

if status["SERPAPI_KEY"]:
    st.sidebar.success("SERP resources enabled")
else:
    st.sidebar.info("SERP resources disabled")
st.sidebar.caption("API keys are read only from environment variables.")

# ----------------------- Header -----------------------------
st.markdown("""
<div class="hero">
  <h1>Career Coach App</h1>
  <p>A role-specific career preparation workspace for ATS analysis, roadmaps, LinkedIn optimization, interview practice, and resume improvement.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='card'><div class='metric-label'>Resume</div><div class='metric-value'>{'Loaded' if st.session_state.resume_text else 'Not loaded'}</div><div class='small-muted'>{st.session_state.resume_name or 'Upload from sidebar'}</div></div>", unsafe_allow_html=True)
with col2:
    if st.session_state.resume_text:
        route = orchestrate_domain(st.session_state.resume_text, target_domain, "general")
        st.session_state.route = route
        route_value = route.primary_domain
    else:
        route_value = target_domain
    st.markdown(f"<div class='card'><div class='metric-label'>Active agent</div><div class='metric-value'>{route_value}</div><div class='small-muted'>Domain-specific evaluator</div></div>", unsafe_allow_html=True)
with col3:
    words = len(st.session_state.resume_text.split()) if st.session_state.resume_text else 0
    st.markdown(f"<div class='card'><div class='metric-label'>Resume text</div><div class='metric-value'>{words} words</div><div class='small-muted'>Shared in current session</div></div>", unsafe_allow_html=True)

if not st.session_state.resume_text:
    st.info("Upload your resume from the sidebar to activate all modules.")

tabs = st.tabs(["Dashboard", "ATS Analyser", "Roadmap", "LinkedIn", "Interview", "Resume"])
llm = LLMClient()

# ----------------------- Dashboard --------------------------
with tabs[0]:
    st.subheader("Dashboard")
    if st.session_state.resume_text:
        route = orchestrate_domain(st.session_state.resume_text, target_domain, "dashboard")
        breakdown = role_fit_breakdown(st.session_state.resume_text, route.primary_domain)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Routing decision")
            st.write(f"Primary agent: **{route.primary_domain}**")
            st.write(f"Secondary context: **{route.secondary_domain or 'None'}**")
            st.write(f"Confidence: **{round(route.confidence*100)}%**")
            st.caption(route.reasoning_summary)
        with c2:
            st.markdown("#### Role-fit snapshot")
            st.metric("Current fit score", f"{breakdown.final_score}/100")
            if breakdown.mismatch_note:
                st.warning(breakdown.mismatch_note)
        st.markdown("#### Evidence")
        ev1, ev2 = st.columns(2)
        with ev1:
            st.write("Strong evidence")
            st.write(", ".join(breakdown.strong_evidence) if breakdown.strong_evidence else "No strong evidence detected yet.")
        with ev2:
            st.write("Missing evidence")
            st.write(", ".join(breakdown.missing_evidence) if breakdown.missing_evidence else "No major missing terms detected.")
    else:
        st.write("Load a resume to view routing and role-fit diagnostics.")

# ----------------------- ATS -------------------------------
with tabs[1]:
    st.subheader("ATS Analyser")
    st.caption("Role-specific ATS analysis with one final score out of 100.")
    jd = st.text_area("Optional job description", height=180, placeholder="Paste the job description here for stricter ATS matching.")
    if st.button("Analyse ATS fit", key="ats_btn"):
        if not st.session_state.resume_text:
            st.error("Upload a resume first.")
        else:
            with st.spinner("Analysing ATS fit..."):
                result = ats_analysis(st.session_state.resume_text, target_domain, jd, llm)
            st.metric("ATS Fit Score", f"{result['score']}/100")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Strong matches")
                st.write(", ".join(result["matched"]) or "None detected")
            with c2:
                st.markdown("#### Missing or weak terms")
                st.write(", ".join(result["missing"]) or "No major gaps detected")
            st.markdown(result["report"])

# ----------------------- Roadmap ---------------------------
with tabs[2]:
    st.subheader("Roadmap Generator")
    c1, c2, c3 = st.columns(3)
    with c1: level = st.selectbox("Current level", ["Beginner", "Intermediate", "Advanced"])
    with c2: duration = st.selectbox("Duration (weeks)", [4, 8, 12, 16], index=1)
    with c3: weekly_hours = st.slider("Weekly hours", 3, 30, 8)
    st.caption(f"Total planned learning time: {duration * weekly_hours} hours over {duration} weeks.")
    preference = st.selectbox("Learning preference", ["Balanced", "Videos", "Documentation", "Practice questions", "Projects"])
    goal = st.selectbox("Goal", ["Placement preparation", "Internship", "Job switch", "Portfolio building", "Interview preparation"])
    if st.button("Generate roadmap", key="roadmap_btn"):
        if not st.session_state.resume_text:
            st.error("Upload a resume first.")
        else:
            with st.spinner("Building roadmap..."):
                result = generate_roadmap(st.session_state.resume_text, target_domain, level, duration, weekly_hours, preference, goal, llm)
            if result["used_live_resources"]:
                st.success("Live resources were included.")
            else:
                st.info("Using curated fallback resources. Add SERPAPI_KEY on the server for live resource search.")
            st.markdown(result["roadmap"])
            if result["resources"]:
                st.markdown("#### Resource links")
                for r in result["resources"]:
                    st.markdown(f"- [{r.get('title','Resource')}]({r.get('link','#')}) — {r.get('snippet','')}")

# ----------------------- LinkedIn --------------------------
with tabs[3]:
    st.subheader("LinkedIn Optimizer")
    linkedin_url = st.text_input("LinkedIn profile URL", placeholder="https://www.linkedin.com/in/your-profile/")
    auto_clicked = st.button("Analyze LinkedIn profile", key="li_analyze")
    if auto_clicked:
        if not st.session_state.resume_text:
            st.error("Upload a resume first.")
        elif not linkedin_url:
            st.error("Enter a LinkedIn profile URL.")
        else:
            with st.spinner("Trying automatic LinkedIn extraction..."):
                extraction = fetch_linkedin_profile_content(linkedin_url)
                st.session_state.linkedin_extraction = extraction
            if extraction.confidence == "LOW":
                st.warning(extraction.message)
            else:
                with st.spinner("Generating LinkedIn optimization..."):
                    result = generate_linkedin_optimization(st.session_state.resume_text, target_domain, linkedin_url, extraction, llm=llm)
                st.info(result.get("message", ""))
                st.markdown(result.get("report", ""))
    extraction = st.session_state.linkedin_extraction
    show_manual = extraction is not None and extraction.confidence == "LOW"
    show_manual = st.checkbox("Optimize selected sections manually", value=show_manual)
    if show_manual:
        st.markdown("Paste only the sections you want improved.")
        headline = st.text_input("Headline")
        about = st.text_area("About", height=120)
        experience = st.text_area("Experience", height=120)
        projects = st.text_area("Projects", height=120)
        skills = st.text_area("Skills", height=80)
        featured = st.text_area("Featured / Certifications", height=80)
        if st.button("Optimize selected sections", key="li_manual"):
            if not st.session_state.resume_text:
                st.error("Upload a resume first.")
            else:
                sections = linkedin_manual_sections(headline=headline, about=about, experience=experience, projects=projects, skills=skills, featured=featured)
                if not sections:
                    st.error("Paste at least one section to optimize.")
                else:
                    extraction = extraction or fetch_linkedin_profile_content(linkedin_url or "https://linkedin.com/in/manual")
                    extraction.content = ""
                    extraction.confidence = "MANUAL"
                    extraction.message = "Manual section optimization was used."
                    with st.spinner("Optimizing selected sections..."):
                        result = generate_linkedin_optimization(st.session_state.resume_text, target_domain, linkedin_url, extraction, sections, llm)
                    st.markdown(result.get("report", ""))

# ----------------------- Interview --------------------------
with tabs[4]:
    st.subheader("Interview Prep Buddy")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.selectbox("Interview section", ["DSA", "ML", "HR", "Resume Deep Dive"])
    with c2:
        start_diff = st.selectbox("Starting difficulty", ["Beginner", "Intermediate", "Advanced"], index=1)
    with c3:
        num_questions = st.slider("Target questions", 3, 15, 6)
    with c4:
        coding_language = st.selectbox("Coding language", ["Python", "C++", "Java"], disabled=(mode != "DSA"))
    col_start, col_reset = st.columns([1, 4])
    with col_start:
        if st.button("Start session", key="start_interview"):
            if mode == "Resume Deep Dive" and not st.session_state.resume_text:
                st.error("Upload a resume for Resume Deep Dive mode.")
            else:
                q = generate_interview_question(mode, target_domain, start_diff, st.session_state.resume_text, [], llm, coding_language=coding_language)
                st.session_state.interview = {"active": True, "mode": mode, "difficulty": start_diff, "questions": [q], "answers": [], "evaluations": [], "current_question": q, "target": num_questions, "coding_language": coding_language}
    with col_reset:
        if st.button("Reset interview", key="reset_interview"):
            st.session_state.interview = {"active": False, "mode": None, "difficulty": "Beginner", "questions": [], "answers": [], "evaluations": [], "current_question": "", "coding_language": "Python"}
    inter = st.session_state.interview
    if inter["active"]:
        lang_note = f" | **Language:** {inter.get('coding_language', coding_language)}" if inter.get("mode") == "DSA" else ""
        st.markdown(f"**Mode:** {inter['mode']} | **Current difficulty:** {inter['difficulty']} | **Question:** {len(inter['questions'])}/{inter.get('target', num_questions)}{lang_note}")
        st.markdown("#### Current question")
        st.write(inter["current_question"])
        answer = st.text_area("Your answer", height=160, key=f"answer_{len(inter['questions'])}_{len(inter['answers'])}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Evaluate answer", key="eval_answer"):
                if not answer.strip():
                    st.error("Write your answer first.")
                else:
                    with st.spinner("Evaluating answer..."):
                        evaluation = evaluate_interview_answer(inter["mode"], target_domain, inter["current_question"], answer, inter["difficulty"], llm, coding_language=inter.get("coding_language", coding_language))
                    inter["answers"].append(answer)
                    inter["evaluations"].append(evaluation)
                    inter["difficulty"] = evaluation["next_difficulty"]
                    st.session_state.interview = inter
        with c2:
            if st.button("Next question", key="next_q"):
                if len(inter["questions"]) >= inter.get("target", num_questions):
                    st.success("Target question count completed. Reset to start a new session.")
                else:
                    with st.spinner("Preparing next adaptive question..."):
                        q = generate_interview_question(inter["mode"], target_domain, inter["difficulty"], st.session_state.resume_text, inter["questions"], llm, coding_language=inter.get("coding_language", coding_language))
                    inter["questions"].append(q)
                    inter["current_question"] = q
                    st.session_state.interview = inter
                    st.rerun()
        if inter["evaluations"]:
            latest = inter["evaluations"][-1]
            st.metric("Latest answer score", f"{latest['score']}/10")
            st.write(latest.get("feedback", ""))
            if latest.get("missing_points"):
                st.write("Missing points:", ", ".join(map(str, latest["missing_points"])))
            with st.expander("Show ideal answer"):
                st.markdown(latest.get("ideal_answer", ""))
    else:
        st.write("Choose an interview section and start a session.")

# ----------------------- Resume -----------------------------
with tabs[5]:
    st.subheader("Resume Improvement")
    st.caption("Strict role-fit review based on the selected target domain.")
    if st.button("Review resume", key="resume_review"):
        if not st.session_state.resume_text:
            st.error("Upload a resume first.")
        else:
            with st.spinner("Reviewing resume..."):
                result = generate_resume_report(st.session_state.resume_text, target_domain, llm)
            br = result["breakdown"]
            st.metric("Resume Fit Score", f"{br.final_score}/100")
            st.write(f"Detected resume direction: **{br.detected_direction}**")
            if br.mismatch_note:
                st.warning(br.mismatch_note)
            st.markdown("#### Score breakdown")
            st.dataframe(pd.DataFrame([{"Component": k, "Score": v} for k, v in br.components.items()]), hide_index=True, use_container_width=True)
            st.markdown(result["analysis"])
