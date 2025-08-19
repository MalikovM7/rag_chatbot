import streamlit as st
import requests
from sseclient import SSEClient
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 RAG Chatbot")

with st.sidebar:
    st.header("Settings")
    use_rag = st.toggle("Use RAG", value=True)
    top_k = st.slider("Top-K", 1, 10, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask something…")
placeholder = st.empty()

def stream_answer(q):
    payload = {"query": q, "use_rag": use_rag, "top_k": top_k, "temperature": temperature}
    url = f"{BACKEND_URL}/chat/stream"
    for event in SSEClient(url, method="POST", json=payload):
        if not event.data:
            continue
        yield event.data

if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    answer_md = ""
    sources_shown = False
    with st.chat_message("assistant"):
        answer_box = st.empty()
        for data in stream_answer(query):
            obj = requests.utils.json.loads(data)
            if obj["type"] == "token":
                if not sources_shown and obj.get("sources"):
                    with st.expander("Retrieved sources"):
                        for s in obj["sources"]:
                            st.write(f"**Score:** {s['score']:.3f}\n\n{s['text']}")
                    sources_shown = True
                answer_md += obj["data"]
                answer_box.markdown(answer_md)
            elif obj["type"] == "done":
                break
    st.session_state.history.append(("assistant", answer_md))
