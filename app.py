import os
import sys
import json
import time
import shutil
import warnings
import subprocess
from typing import Dict, TypedDict

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["STREAMLIT_LOG_LEVEL"] = "error"

from dotenv import load_dotenv
import streamlit as st

load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Multi-Agent AI Engineering Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stCodeBlock { border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; padding: 8px 16px; }
    </style>
""", unsafe_allow_html=True)


class AgentTeamState(TypedDict):
    user_prompt: str
    architecture: str
    files: Dict[str, str]
    test_results: str
    test_passed: bool
    review_comments: str
    documentation: str
    debug_retries: int


# ---------------------------------------------------------------------
# Fallback code (only used if the AI call fails for every candidate model)
# ---------------------------------------------------------------------
FALLBACK_APP_PY = """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI(title="Employee Management REST API")
employees_db: Dict[int, dict] = {}

class Employee(BaseModel):
    id: int
    name: str
    role: str
    department: str

class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None

@app.post("/employees/", status_code=201)
def create_employee(emp: Employee):
    if emp.id in employees_db:
        raise HTTPException(status_code=400, detail="Employee ID already exists.")
    employees_db[emp.id] = emp.dict()
    return employees_db[emp.id]

@app.get("/employees/", response_model=List[dict])
def list_employees():
    return list(employees_db.values())

@app.get("/employees/{emp_id}")
def get_employee(emp_id: int):
    if emp_id not in employees_db:
        raise HTTPException(status_code=404, detail="Employee not found.")
    return employees_db[emp_id]

@app.put("/employees/{emp_id}")
def update_employee(emp_id: int, emp_data: EmployeeUpdate):
    if emp_id not in employees_db:
        raise HTTPException(status_code=404, detail="Employee not found.")
    current_data = employees_db[emp_id]
    updated_data = emp_data.dict(exclude_unset=True)
    current_data.update(updated_data)
    employees_db[emp_id] = current_data
    return current_data

@app.delete("/employees/{emp_id}")
def delete_employee(emp_id: int):
    if emp_id not in employees_db:
        raise HTTPException(status_code=404, detail="Employee not found.")
    deleted_emp = employees_db.pop(emp_id)
    return {"message": "Employee deleted successfully", "employee": deleted_emp}
"""

FALLBACK_TEST_APP_PY = """from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_full_employee_crud():
    res_create = client.post("/employees/", json={"id": 101, "name": "Sarah Khan", "role": "Senior Dev", "department": "Engineering"})
    assert res_create.status_code == 201
    assert res_create.json()["name"] == "Sarah Khan"

    res_get = client.get("/employees/101")
    assert res_get.status_code == 200

    res_update = client.put("/employees/101", json={"role": "Lead Architect"})
    assert res_update.status_code == 200

    res_list = client.get("/employees/")
    assert res_list.status_code == 200

    res_delete = client.delete("/employees/101")
    assert res_delete.status_code == 200
"""


# ---------------------------------------------------------------------
# LLM setup — FIXED: actually tests each candidate model with a real
# call instead of just constructing the object, and caches the working
# model in session_state so we don't re-test it on every single call.
# ---------------------------------------------------------------------
def get_llm_instance():
    if "llm_instance" in st.session_state:
        return st.session_state["llm_instance"]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.session_state["llm_instance"] = (None, "No API Key")
        return None, "No API Key"

    candidate_models = [
        "gemini-3.6-flash",       # stable, much higher free daily quota
        "gemini-3.5-flash",       # stable fallback
        "gemini-3.1-flash-lite",  # lightweight fallback
        "gemini-flash-latest",    # newest model — kept last: often has a very
                                   # low free daily quota (e.g. 20/day) since
                                   # it's the newest/preview release
    ]

    for model_name in candidate_models:
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,
                google_api_key=api_key,
                timeout=90,  # FIXED: 15s was enough for the "ping" test but
                             # too short for real generation (architecture,
                             # full code files), causing 504 DEADLINE_EXCEEDED
            )
            # This is the important fix: constructing the object never
            # fails, so we have to actually call it to know if the model
            # name / key / quota is valid.
            model.invoke([HumanMessage(content="ping")])
            print(f"[LLM] Using model: {model_name}")
            st.session_state["llm_instance"] = (model, model_name)
            return model, model_name
        except Exception as e:
            # FIXED: log the real error instead of silently swallowing it
            print(f"[LLM ERROR - {model_name}]: {e}")
            continue

    st.session_state["llm_instance"] = (None, "No working model found")
    return None, "No working model found"


def safe_llm_invoke(prompt: str, fallback_text: str) -> str:
    """For agents that return plain markdown/text (Manager, Reviewer, Docs)."""
    llm, model_name = get_llm_instance()
    if llm is not None:
        try:
            time.sleep(1)
            res = llm.invoke([HumanMessage(content=prompt)])
            return str(res.content)
        except Exception as e:
            print(f"[LLM CALL ERROR - {model_name}]: {e}")
    return f"⚠️ **Fallback Active (Offline / Quota Exceeded Mode)**\n\n{fallback_text}"


def safe_llm_invoke_json(prompt: str, fallback_files: Dict[str, str]) -> Dict[str, str]:
    """For agents that must return a dict of {filename: content} (Developer, Debugger).
    Returns fallback_files directly on failure — never mixes banner text into the JSON."""
    llm, model_name = get_llm_instance()
    if llm is not None:
        try:
            time.sleep(1)
            res = llm.invoke([HumanMessage(content=prompt)])
            raw = str(res.content).strip()
            for tag in ["```json", "```"]:
                raw = raw.replace(tag, "")
            raw = raw.strip()
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed:
                return parsed
            print(f"[LLM JSON WARNING - {model_name}]: parsed but empty/wrong shape")
        except Exception as e:
            print(f"[LLM JSON ERROR - {model_name}]: {e}")
    return fallback_files


# --- AGENT FUNCTIONS ---
def manager_agent(state: AgentTeamState) -> Dict:
    prompt = f"Design complete system architecture and detailed endpoints for: {state['user_prompt']}"
    fallback = (
        "### System Architecture (REST API)\n"
        "- **Base Endpoint**: `/api/v1`\n"
        "- **Endpoints Spec**:\n"
        "  - `POST /employees/` - Create new employee record\n"
        "  - `GET /employees/` - Retrieve all employee records\n"
        "  - `GET /employees/{emp_id}` - Retrieve specific employee details\n"
        "  - `PUT /employees/{emp_id}` - Update employee record\n"
        "  - `DELETE /employees/{emp_id}` - Remove employee record\n"
    )
    arch = safe_llm_invoke(prompt, fallback)
    return {"architecture": arch, "debug_retries": 0}


def developer_agent(state: AgentTeamState) -> Dict:
    # FIXED: this agent now actually calls the AI instead of returning
    # the same hardcoded FastAPI code every single run.
    prompt = (
        "You are a backend developer. Based on this architecture, generate a "
        "complete, working FastAPI application.\n\n"
        f"Architecture:\n{state['architecture']}\n\n"
        "Respond with ONLY valid JSON, no markdown fences, no explanation, "
        "in this exact shape:\n"
        '{"app.py": "<full file content as a string>", '
        '"test_app.py": "<full pytest file content as a string>"}\n\n'
        "Requirements:\n"
        "- The FastAPI app object must be named `app`, importable as `from app import app`.\n"
        "- Use an in-memory dict for storage (no database needed).\n"
        "- Implement full CRUD matching the architecture.\n"
        "- test_app.py must use fastapi.testclient.TestClient and cover "
        "create/list/get/update/delete."
    )
    fallback_files = {
        "app.py": FALLBACK_APP_PY,
        "test_app.py": FALLBACK_TEST_APP_PY,
    }
    files = safe_llm_invoke_json(prompt, fallback_files)
    return {"files": files}


def sandbox_tester_agent(state: AgentTeamState) -> Dict:
    workspace_dir = os.path.abspath("./sandbox_workspace")
    os.makedirs(workspace_dir, exist_ok=True)

    for filename, code in state["files"].items():
        with open(os.path.join(workspace_dir, filename), "w", encoding="utf-8") as f:
            f.write(code)

    print(f"[SANDBOX] Running pytest in {workspace_dir} ...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "test_app.py"],
            cwd=workspace_dir, capture_output=True, text=True,
            timeout=60,  # FIXED: without this, a hung test = a hung app forever
        )
        print(f"[SANDBOX] pytest finished with exit code {result.returncode}")
        if result.returncode == 0:
            return {"test_results": result.stdout, "test_passed": True}
        else:
            return {"test_results": result.stdout + "\n" + result.stderr, "test_passed": False}
    except subprocess.TimeoutExpired:
        print("[SANDBOX ERROR]: pytest timed out after 60 seconds")
        return {
            "test_results": "Sandbox test run timed out after 60 seconds. "
                             "The generated code may contain a blocking call "
                             "(e.g. uvicorn.run() at import time, or an infinite loop).",
            "test_passed": False,
        }
    except Exception as err:
        print(f"[SANDBOX ERROR]: {err}")
        return {"test_results": str(err), "test_passed": False}


def debugger_agent(state: AgentTeamState) -> Dict:
    retries = state.get("debug_retries", 0) + 1
    prompt = (
        f"The following pytest run failed:\n{state['test_results']}\n\n"
        f"Current files:\n{json.dumps(state['files'])}\n\n"
        "Fix the bug(s). Respond with ONLY valid JSON with the corrected full "
        'file contents, shape: {"app.py": "...", "test_app.py": "..."}. '
        "Only include files you actually changed, but each included file must "
        "be the COMPLETE corrected content, not a diff."
    )
    fixed = safe_llm_invoke_json(prompt, {})
    if fixed:
        updated_files = {**state["files"], **fixed}
        return {"files": updated_files, "debug_retries": retries}
    return {"debug_retries": retries}


def code_reviewer_agent(state: AgentTeamState) -> Dict:
    prompt = f"Perform code review and security assessment for:\n{state['files']}"
    fallback = (
        "### Security & Quality Assessment\n"
        "- **Validation**: Pydantic schemas properly enforce data types.\n"
        "- **HTTP Status Codes**: Explicit error handling for 400 and 404 conditions.\n"
        "- **Persistence**: Currently using in-memory dict. Migrate to PostgreSQL/SQLite for production."
    )
    review = safe_llm_invoke(prompt, fallback)
    return {"review_comments": review}


def docs_and_git_agent(state: AgentTeamState) -> Dict:
    readme_prompt = f"Generate a comprehensive README.md documentation for:\n{state['files']}"
    fallback = (
        "# Employee Management REST API\n"
        "Production-grade FastAPI application with automated Pytest suite.\n\n"
        "## Local Execution Instructions\n"
        "1. Install dependencies: `pip install fastapi uvicorn pytest httpx`\n"
        "2. Run server: `uvicorn app:app --reload`\n"
        "3. Run automated test suite: `pytest -v test_app.py`\n"
    )
    docs = safe_llm_invoke(readme_prompt, fallback)

    workspace_dir = os.path.abspath("./sandbox_workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    with open(os.path.join(workspace_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(docs)

    # FIXED: check whether git is even installed before trying to use it,
    # instead of always hitting the except branch on machines without git.
    if shutil.which("git") is None:
        git_status = (
            "⚠️ Git is not installed on this system — commit skipped. "
            "Install Git (git-scm.com) to enable automatic version control."
        )
    else:
        try:
            subprocess.run(["git", "init"], cwd=workspace_dir, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=workspace_dir, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "feat: AI generated REST API"],
                cwd=workspace_dir, check=True, capture_output=True
            )
            git_status = "✅ Git repository initialized & committed successfully."
        except Exception as e:
            print(f"[GIT ERROR]: {e}")
            git_status = f"⚠️ Git status: {str(e)}"

    return {"documentation": f"{docs}\n\n**Git Commit Log:** {git_status}"}


def should_debug(state: AgentTeamState):
    if not state["test_passed"] and state.get("debug_retries", 0) < 3:
        return "debugger"
    return "code_reviewer"


# --- SIDEBAR UI ---
st.sidebar.title("⚙️ Control Panel")

api_key_check = os.getenv("GOOGLE_API_KEY")
if api_key_check:
    st.sidebar.success("🔑 API Key Detected in .env")
else:
    st.sidebar.warning("⚠️ No GOOGLE_API_KEY found in .env (Offline Mode)")

if st.sidebar.button("🔄 Reset cached AI model"):
    st.session_state.pop("llm_instance", None)
    st.sidebar.info("Cache cleared — next run will re-test all models.")

st.sidebar.markdown("### 👥 Agent Roster")
st.sidebar.markdown("""
* 👔 **Manager Agent**: System Design
* 💻 **Developer Agent**: FastAPI Backend (real AI)
* 🧪 **Sandbox Tester**: Pytest Suite
* 🛠️ **Debugger Agent**: Patch Repair (real AI)
* 🔍 **Reviewer Agent**: Security Audit
* 📚 **Docs & Git Agent**: Commit & Docs
""")

# --- MAIN UI ---
st.title("🤖 Multi-Agent AI Engineering Dashboard")
st.markdown("Automated FastAPI Architecture, Code Generation, Sandbox Testing, and Git Versioning Pipeline")

prompt_input = st.text_area(
    "Project Requirements:",
    value="Build a production-grade REST API for an Employee Management System with Full CRUD functionality.",
    height=80
)

if st.button("🚀 Run Multi-Agent Pipeline", type="primary", use_container_width=True):

    workflow = StateGraph(AgentTeamState)
    workflow.add_node("manager", manager_agent)
    workflow.add_node("developer", developer_agent)
    workflow.add_node("tester", sandbox_tester_agent)
    workflow.add_node("debugger", debugger_agent)
    workflow.add_node("code_reviewer", code_reviewer_agent)
    workflow.add_node("documentation", docs_and_git_agent)

    workflow.set_entry_point("manager")
    workflow.add_edge("manager", "developer")
    workflow.add_edge("developer", "tester")
    workflow.add_conditional_edges("tester", should_debug, {"debugger": "debugger", "code_reviewer": "code_reviewer"})
    workflow.add_edge("debugger", "tester")
    workflow.add_edge("code_reviewer", "documentation")
    workflow.add_edge("documentation", END)

    app_graph = workflow.compile()

    # --- LIVE AGENT TRACKER ---
    status_box = st.empty()

    status_box.info("👔 **Manager Agent**: System architecture design kar raha hai...")
    time.sleep(1)

    status_box.info("💻 **Developer Agent**: FastAPI code aur Pytest cases generate kar raha hai...")
    time.sleep(1)

    status_box.info("🧪 **Sandbox Tester Agent**: `./sandbox_workspace` mein Pytest suite execute kar raha hai...")

    final_output = app_graph.invoke({"user_prompt": prompt_input})

    status_box.info("🔍 **Reviewer & Docs Agents**: Security assessment, README.md, aur Git commit final kar rahe hain...")
    time.sleep(1)

    status_box.empty()
    st.success("🎉 Multi-Agent Pipeline Executed Successfully!")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗️ Architecture",
        "💻 Generated Code",
        "🧪 Test Results",
        "🔍 Code Review",
        "📚 Docs & Git"
    ])

    with tab1:
        st.subheader("System Architecture & API Design")
        st.write(final_output.get("architecture"))

    with tab2:
        st.subheader("FastAPI Application & Unit Tests")
        for fname, code in final_output.get("files", {}).items():
            st.markdown(f"📄 **`{fname}`**")
            st.code(code, language="python")

    with tab3:
        st.subheader("Sandbox Execution Output")
        st.code(final_output.get("test_results"), language="bash")

    with tab4:
        st.subheader("Security & Performance Assessment")
        st.write(final_output.get("review_comments"))

    with tab5:
        st.subheader("Documentation & Version Control")
        st.write(final_output.get("documentation"))