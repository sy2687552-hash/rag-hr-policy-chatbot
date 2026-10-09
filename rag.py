"""Lightweight, token-efficient RAG engine for HR policy documents.

Pipeline: parse -> chunk by heading -> TF-IDF index -> retrieve top-k ->
(optional) LLM answer from ONLY those chunks. Without an API key it falls
back to an extractive answer (zero tokens)."""
import json, os, re, hashlib, urllib.request
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")  # cheapest tier
TOP_K, MAX_CTX_CHARS, MAX_OUT_TOKENS, MIN_SCORE = 3, 2600, 350, 0.07
SYSTEM = ("You are the HR policy assistant for NexaSphere Technologies. Answer ONLY from the provided "
          "policy excerpts. Be concise (max 120 words), use short bullets if helpful, and name the "
          "relevant section. If the excerpts don't answer the question, say it isn't specified in "
          "the policy and suggest contacting HR. Never invent numbers or rules.")
STOP = set("a an the is are am to of in on for and or my me i do does can we you what how when who which it be if with at by from as that this there any about".split())


# ---------- parsing ----------
def read_file(path):
    """Return list of (heading, text) sections."""
    p = Path(path); ext = p.suffix.lower()
    if ext == ".docx":
        import docx
        d, secs, head, buf = docx.Document(p), [], "Introduction", []
        for para in d.paragraphs:
            t = para.text.strip()
            if not t: continue
            if para.style.name.lower().startswith("heading"):
                if buf: secs.append((head, "\n".join(buf)))
                head, buf = t, []
            else: buf.append(t)
        if buf: secs.append((head, "\n".join(buf)))
        return secs
    if ext == ".pdf":
        from pypdf import PdfReader
        return [(f"Page {i+1}", (pg.extract_text() or "")) for i, pg in enumerate(PdfReader(p).pages)]
    txt = p.read_text(errors="ignore")
    parts = re.split(r"(?m)^#+\s*(.+)$", txt)
    if len(parts) > 1:
        return [(parts[i], parts[i+1].strip()) for i in range(1, len(parts), 2)]
    return [("Document", txt)]


def chunk(sections, size=900):
    out = []
    for head, text in sections:
        text = re.sub(r"\n\s*Owner: .*?material change\s*", "\n", text)  # strip boilerplate footer
        if len(text.strip()) < 40: continue
        buf = ""
        for para in re.split(r"\n+", text):
            if len(buf) + len(para) > size and buf:
                out.append({"title": head, "text": buf.strip()}); buf = ""
            buf += para + "\n"
        if buf.strip(): out.append({"title": head, "text": buf.strip()})
    return out


# ---------- index ----------
class Index:
    def __init__(self):
        self.chunks, self.cache, self.source = [], {}, None

    def build(self, path):
        self.chunks = chunk(read_file(path)); self.source = Path(path).name; self.cache = {}
        docs = [f"{c['title']} {c['title']} {c['text']}" for c in self.chunks]  # title boost
        import numpy as np
        self.boost = np.array([1.5 if self.is_hr(c["title"]) else 1.0 for c in self.chunks])
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=list(STOP), sublinear_tf=True)
        self.mat = self.vec.fit_transform(docs)
        return len(self.chunks)

    def search(self, q, k=TOP_K):
        sims = linear_kernel(self.vec.transform([q]), self.mat).ravel() * self.boost  # prefer HR sections
        idx = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in idx if sims[i] >= MIN_SCORE]

    @staticmethod
    def is_hr(title):
        """HR sections: pages 011-022 of the pack and the numbered HR manual (appendix)."""
        m = re.match(r"(\d{3})\.", title)
        if m: return 11 <= int(m.group(1)) <= 22
        return bool(re.match(r"\d{1,2}\.", title)) or "HR" in title

    # ---------- answering ----------
    def answer(self, q):
        key = hashlib.md5(q.lower().strip().encode()).hexdigest()
        if key in self.cache: return {**self.cache[key], "cached": True}
        hits = self.search(q)
        if not hits:
            res = {"answer": "I couldn't find this in the HR policy document. Please contact the HR team directly for help.",
                   "sources": [], "mode": "none"}
            return res
        ctx, used = "", []
        for c, s in hits:
            piece = f"[{c['title']}]\n{c['text']}\n\n"
            if len(ctx) + len(piece) > MAX_CTX_CHARS: break
            ctx += piece; used.append((c, s))
        sources = [{"title": c["title"], "score": round(s, 2), "text": c["text"][:420]} for c, s in used]
        key_api = os.getenv("ANTHROPIC_API_KEY")
        try:
            gkey = os.getenv("GEMINI_API_KEY")
            if gkey: ans, mode = self._gemini(q, ctx, gkey), "llm (gemini)"
            elif key_api: ans, mode = self._llm(q, ctx, key_api), "llm"
            else: ans, mode = self._extract(q, used), "extractive"
        except Exception as e:
            ans, mode = self._extract(q, used), f"extractive - AI error: {str(e)[:300]}"
        res = {"answer": ans, "sources": sources, "mode": mode}
        self.cache[key] = res
        return res

    def _gemini(self, q, ctx, key):
        import urllib.error
        models = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"] if m]
        body = {"system_instruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"parts": [{"text": f"Policy excerpts:\n{ctx}\nQuestion: {q}"}]}],
                "generationConfig": {"maxOutputTokens": 700, "temperature": 0.2}}
        last = "no model tried"
        for model in models:
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                json.dumps(body).encode(), {"content-type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.load(r)
                parts = data["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts).strip()
                if text: return text
                last = f"{model}: empty reply"
            except urllib.error.HTTPError as e:
                last = f"{model}: HTTP {e.code} {e.read().decode()[:200]}"
                if e.code in (400, 401, 403) and "API key" in last: break   # bad key: other models won't help
            except Exception as e:
                last = f"{model}: {type(e).__name__} {e}"
        raise RuntimeError(last)

    def _llm(self, q, ctx, key):
        body = {"model": MODEL, "max_tokens": MAX_OUT_TOKENS,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": f"Policy excerpts:\n{ctx}\nQuestion: {q}"}]}
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", json.dumps(body).encode(),
              {"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.load(r)["content"][0]["text"].strip()

    def _extract(self, q, used):
        terms = {w for w in re.findall(r"\w+", q.lower()) if w not in STOP}
        scored = []
        for c, _ in used:
            for s in re.split(r"(?<=[.!?])\s+|\n", c["text"]):
                s = s.strip()
                if len(s) < 25: continue
                sc = len(terms & set(re.findall(r"\w+", s.lower())))
                if sc: scored.append((sc, s, c["title"]))
        scored.sort(key=lambda x: -x[0])
        top = scored[:4]
        if not top: return used[0][0]["text"][:400]
        return "\n".join(f"• {s}" for _, s, _ in top) + f"\n\nSection: {top[0][2]}"
