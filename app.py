import os, tempfile
from flask import Flask, request, jsonify, send_from_directory
from rag import Index

def _load_env(path):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
_load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"))
idx = Index()
DEFAULT = os.path.join(BASE, "data", "fictional_company_demo_data_with_detailed_hr_policy.docx")
if os.path.exists(DEFAULT): idx.build(DEFAULT)

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Cache-Control"] = "no-store"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return r

@app.route("/api/<path:_>", methods=["OPTIONS"])
def preflight(_): return ("", 204)

@app.errorhandler(404)
def nf(e): return jsonify(error=f"Not found: {request.method} {request.path}", routes=sorted(r.rule for r in app.url_map.iter_rules())), 404

@app.get("/")
def home(): return send_from_directory(os.path.join(BASE, "static"), "index.html")

@app.get("/api/status")
def status():
    return jsonify(version="v6", source=idx.source, chunks=len(idx.chunks), llm=bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")))

@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith((".pdf", ".docx", ".txt", ".md")):
        return jsonify(error="Upload a PDF, DOCX, TXT or MD file"), 400
    path = os.path.join(tempfile.gettempdir(), f.filename); f.save(path)
    try: n = idx.build(path)
    except Exception as e: return jsonify(error=f"Could not read file: {e}"), 400
    idx.source = f.filename
    return jsonify(source=f.filename, chunks=n)

@app.post("/api/ask")
def ask():
    q = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not q: return jsonify(error="Empty question"), 400
    if not idx.chunks: return jsonify(error="Upload a policy document first"), 400
    return jsonify(idx.answer(q[:500]))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"\n  NexaSphere HR Assistant running -> http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
