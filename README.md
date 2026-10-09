# 🤖 NexaSphere HR Policy Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers employee questions
directly from the company HR policy, so nobody has to read a 40-page document.

> NexaSphere Technologies is a fictional company created for demo purposes.

<!-- Add a screenshot: ![Screenshot](screenshot.png) -->

## ❓ The problem
Employees rarely read HR policies, so HR answers the same questions repeatedly
(leave, work from home, probation, grievances, notice period).

## ✅ The solution
Employees ask a question in plain language. The app finds the most relevant
policy sections and returns a short, grounded answer, with the source shown.

## ✨ Features
- Answers based only on the policy document
- Source panel under every answer
- Refuses out-of-scope questions without calling the model
- Works with Gemini or Claude, plus an offline extractive mode (no API key)
- Responsive UI with dark mode and suggested questions
- Two ways to run: Flask backend (PDF/DOCX/TXT upload) or a standalone HTML file

## 🧠 How it works
1. **Ingest:** parse the policy and split by heading into ~900-character sections
2. **Index:** TF-IDF vectors (unigrams + bigrams), headings weighted higher
3. **Retrieve:** cosine similarity selects the top 3 sections (HR sections boosted)
4. **Guard:** low-similarity questions are rejected without an LLM call
5. **Generate:** the LLM answers using only the retrieved sections
6. **Display:** answer and sources are shown

## 💸 Token efficiency
- Local retrieval, no embedding API calls
- Only the top 3 sections (~2,600 characters) sent to the model
- Capped output, lightweight model by default, cached repeat questions

## 🛠 Tech stack
Python · Flask · scikit-learn · python-docx · pypdf · JavaScript · HTML/CSS · Gemini API / Claude API

## 🚀 Run it

**Option 1: Standalone (no install).** Open `NexaSphere-HR-Assistant.html` in your browser.
Optionally paste a Gemini API key in the sidebar (stored only in your browser).

**Option 2: Flask version**
```bash
pip install -r requirements.txt
cp .env.example .env        # add your GEMINI_API_KEY (or ANTHROPIC_API_KEY)
python app.py               # open http://127.0.0.1:8000
```
On Windows you can double-click `run.bat`; on Mac/Linux run `./run.sh`.

## 📁 Project structure
```
├── app.py                          # Flask server and API routes
├── rag.py                          # chunking, TF-IDF retrieval, LLM calls
├── static/index.html               # UI for the Flask version
├── data/                           # sample HR policy (fictional)
├── NexaSphere-HR-Assistant.html    # standalone version, no server needed
├── requirements.txt
└── .env.example
```

## ⚠️ Limitations
- TF-IDF keyword retrieval, not neural embeddings, so very differently worded questions may match poorly
- The standalone version loads only .txt/.md files
- Answer quality depends on how specific the source policy is

## 🔮 Future improvements
- Embeddings with a vector store (FAISS/Chroma) and a re-ranker
- Evaluation on a labelled question set
- Login, admin policy management, chat history

## 🔒 Security
Never commit your `.env` file or API keys.

## 📄 License
MIT
