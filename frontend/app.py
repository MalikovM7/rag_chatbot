import json
import os
from typing import Any, Dict, List

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HISTORY_FILE = "/tmp/chat_history.json"

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 RAG Chatbot (Bedrock)")

with st.sidebar:
    st.header("Settings")
    use_rag = st.toggle("Use RAG", value=True, help="Let the model call the retriever tool when helpful.")
    top_k = st.slider("Top-K documents", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.number_input("Max tokens (model output)", min_value=64, max_value=4096, value=512, step=64)
    st.caption(f"Backend: {BACKEND_URL}")

if "history" not in st.session_state:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
            st.session_state.history = json.load(fh)
    except Exception:
        st.session_state.history = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

def _save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(st.session_state.history, fh, ensure_ascii=False)
    except Exception:
        pass

for role, msg in st.session_state.history:
    with st.chat_message(role):
        if isinstance(msg, dict) and msg.get("__type__") == "assistant_chunk":
            if msg.get("sources"):
                with st.expander("Sources (RAG)", expanded=False):
                    for i, s in enumerate(msg["sources"]):
                        st.markdown(f"**{i+1}.** {s.get('title') or 'Source'}")
                        text = s.get("text", "")
                        st.code(text[:1200] + ("..." if len(text) > 1200 else ""))
            st.markdown(msg.get("text", ""))
        else:
            st.markdown(msg)

prompt = st.chat_input("Ask me anything…")
if prompt:
    st.session_state.history.append(("user", prompt))
    _save_history()
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        box = st.empty()
        rag_box = st.empty()
        out = ""
        got_any_token = False

        try:
            url = f"{BACKEND_URL}/chat/stream"
            payload = {
                "message": prompt,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "use_rag": bool(use_rag),
                "top_k": int(top_k),
            }

            with requests.post(
                url,
                json=payload,
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=300,
            ) as r:
                r.raise_for_status()
                sources_buf: List[Dict[str, Any]] = []

                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith(":"):
                        continue

                    # NEEDS TO BE FIXED
                    line = raw[5:].strip() if raw.startswith("data:") else raw.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except Exception:
                        out += f"\n\n[debug] {line}"
                        box.markdown(out)
                        continue

                    t = obj.get("type")
                    if t == "sources":
                        sources_buf = obj.get("sources", []) or []
                        with rag_box.container():
                            st.info("Retrieved context (RAG)")
                            for i, s in enumerate(sources_buf):
                                with st.expander(f"Source {i+1}: {s.get('title') or s.get('id')}", expanded=False):
                                    st.code(s.get("text", ""))
                    elif t == "token":
                        got_any_token = True
                        out += obj.get("data", "")
                        if out.endswith((".", "!", "?")):
                            box.markdown(out.strip())
                    elif t == "error":
                        got_any_token = True
                        out += f"\n\n**[backend error]** {obj.get('data')}"
                        box.markdown(out)
                    elif t == "done":
                        break

            if got_any_token and out and not out.endswith((".", "!", "?")):
                box.markdown(out.strip())

            st.session_state.history.append((
                "assistant",
                {"__type__": "assistant_chunk", "text": out.strip(), "sources": st.session_state.last_sources or []}
            ))
            st.session_state.last_sources = []
            _save_history()

        except Exception as ex:
            out = f"Streaming error: {ex}"
            st.error(out)
            st.session_state.history.append(("assistant", out))
            _save_history()

st.caption("Tip: toggle RAG off to compare answers with and without retrieved context.")
