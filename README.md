### 🤖 Multi-Agent AI

Ye ek Multi-Agent AI hai jo [yahan 1 line me batao kya karta hai]. 
Example: "multiple AI agents ko saath me kaam karwata hai"

## 🚀 Kaise Use Karein
```bash
pip install -r requirements.txt
python app.py 🤖 Multi-Agent AI Engineering Dashboard
**Tagline:** LangGraph + Gemini + Streamlit se automated code generation pipeline
An AI-powered system that automates the entire software engineering workflow using 4 collaborating LLM agents.
## 🚀 Live Demo
[https://my-app-cxwacx8turf2yhvoransvv.streamlit.app/](https://my-app-cxwacx8turf2yhvoransvv.streamlit.app/)
## 📸 Dashboard & Agents Running
<img width="960" height="510" alt="agent" src="https://github.com/user-attachments/assets/8134fd26-fedf-40b1-a897-6a34f9b275c7" />
<img width="960" height="510" alt="agent 2" src="https://github.com/user-attachments/assets/5ca76366-71e1-4c8e-aac1-fa1b78cf954b" />
<img width="960" height="510" alt="agent 1" src="https://github.com/user-attachments/assets/0defcebb-db41-4f2c-92d9-999b03ba2463" /
System Architecture
The system has 4 agents:
1.  **Manager Agent**: Creates system design from user prompt
2.  **Developer Agent**: Generates FastAPI code automatically  
3.  **Tester Agent**: Runs code in sandbox and catches errors
4.  **Debugger Agent**: Fixes bugs automatically - fallback active
 🛠️ Tech Stack
`Python` `LangGraph` `LangChain` `Google Gemini 1.5 Flash` `Streamlit` `GitHub` `Streamlit Cloud` `FastAPI`
 Project Structure
sandbox_workspace/
├── app.py                    # Streamlit Dashboard
├── multi_agent_pipeline.py   # LangGraph Agents Logic
├── requirements.txt          # Dependencies
└── sandbox/                  # Code execution environment
 How it Works
1. Enter a prompt like "Build a todo app"
2. Manager creates the system design
3. Developer writes the code
4. Tester runs it and Debugger fixes bugs
5. Get final working code + logs
How to Run Locally
1. Clone repo
   ```bash
   git clone https://github.com/HAS-hub-bot/sandbox_workspace.git
2. Install dependencies
   pip install -r requirements.txt
3. Add Gemini API Key in `.env`
4. Run app
   streamlit run app.py
What I Learned
- Building Agentic AI systems with LangGraph
- Multi-agent orchestration and prompt engineering
- CI/CD deployment with Streamlit Cloud
- End-to-end application development
 Links
- *Live Demo*: https://my-app-cxwacx8turf2yhvoransvv.streamlit.app/
- *GitHub Repo*: https://github.com/HAS-hub-bot/sandbox_workspace
- Built by: *HAS-hub-bot*  
#AI #LangGraph #Streamlit #MachineLearning #OpenToWork
stars dena nahi bholy
