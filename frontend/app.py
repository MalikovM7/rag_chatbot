
import os
import json
import requests
import streamlit as st
from typing import List, Dict, Any

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 RAG Chatbot (Bedrock)")


with st.sidebar:
    st.header("Settings")
    use_rag = st.toggle("Use RAG", value=True, help="Let the model call the retriever tool when helpful.")
    top_k = st.slider("Top-K documents", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.number_input("Max tokens (model output)", min_value=64, max_value=4096, value=512, step=64)
    st.caption("Backend: {}".format(BACKEND_URL))

if "history" not in st.session_state:
    st.session_state.history = []  
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []  


for role, msg in st.session_state.history:
    with st.chat_message(role):
        if isinstance(msg, dict) and msg.get("__type__") == "assistant_chunk":
           
            if msg.get("sources"):
                with st.expander("Sources (RAG)", expanded=False):
                    for i, s in enumerate(msg["sources"]):
                        st.markdown(f"**{i+1}.** {s.get('title') or 'Source'}")
                        st.code(s.get('text','')[:1200] + ("..." if len(s.get('text',''))>1200 else ""))
            st.markdown(msg.get("text",""))
        else:
            st.markdown(msg)

prompt = st.chat_input("Ask me anything…")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        box = st.empty()
        rag_box = st.empty()
        out = ""

        try:
            url = f"{BACKEND_URL}/chat/stream"
            payload = {
                "message": prompt,     
                "temperature": 0.3,
                "max_tokens": 512,
                "use_rag": True,
                "top_k": 4
}

            with requests.post(url, json=payload, stream=True, headers={"Accept": "text/event-stream"}, timeout=120) as r:
                r.raise_for_status()
                sources_buf: List[Dict[str,Any]] = []
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith("data:"):
                        line = raw[len("data:"):].strip()
                    else:
                        
                        line = raw.strip()
                    #NEEDS TO BE FIXED

                    try:
                        obj = json.loads(line)
                    except Exception:
                        
                        out += f"\n\n[debug] {line}"
                        box.markdown(out)
                        continue
                    #---------
                    t = obj.get("type")
                    if t == "sources":
                        sources_buf = obj.get("sources", [])
                        with rag_box.container():
                            st.info("Retrieved context (RAG)")
                            for i, s in enumerate(sources_buf):
                                with st.expander(f"Source {i+1}: {s.get('title') or s.get('id')}", expanded=False):
                                    st.code(s.get("text",""))
                    elif t == "token":
                        got_any_token = True
                        out += obj.get("data", "")
                        if out.endswith((".", "!", "?")):
                            box.markdown(out.strip())
                    elif t == "error":
                        out += f"\n\n**[backend error]** {obj.get('data')}"
                        box.markdown(out)
                    elif t == "done":
                        
                        st.session_state.history.append(("assistant", {"__type__": "assistant_chunk", "text": out, "sources": sources_buf}))
                        st.session_state.last_sources = sources_buf
                        break
        except Exception as ex:
            out = f"Streaming error: {ex}"
            st.error(out)
            st.session_state.history.append(("assistant", out))


st.caption("Tip: toggle RAG off to compare answers with and without retrieved context.")
