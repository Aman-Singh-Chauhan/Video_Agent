"""
Streamlit UI for the AI Video Assistant pipeline.

Run with:
    streamlit run app.py

Expects the same project structure as your CLI script, i.e. these modules
importable from the app's working directory:
    utils.audio_processor.process_input
    core.transcriber.transcribe_all
    core.summarizer.summarize, generate_title
    core.extractor.extract_action_items, extract_key_decisions, extract_questions
    core.rag_engine.build_rag_chain, ask_question
"""

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Session state init
# --------------------------------------------------------------------------- #
defaults = {
    "result": None,
    "chat_history": [],   # list of (role, message) tuples
    "processing": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_state():
    st.session_state.result = None
    st.session_state.chat_history = []


# --------------------------------------------------------------------------- #
# Pipeline runner (mirrors run_pipeline from the CLI script, with progress UI)
# --------------------------------------------------------------------------- #
def run_pipeline_with_progress(source: str, language: str) -> dict:
    progress = st.progress(0, text="Starting AI Video Assistant…")
    status = st.empty()

    status.info("🔊 Processing input & chunking audio…")
    chunks = process_input(source)
    progress.progress(15, text="Audio processed")

    status.info("📝 Transcribing…")
    transcript = transcribe_all(chunks, language)
    progress.progress(40, text="Transcription complete")

    status.info("🏷️ Generating title…")
    title = generate_title(transcript)
    progress.progress(50, text="Title generated")

    status.info("📋 Summarizing…")
    summary = summarize(transcript)
    progress.progress(65, text="Summary generated")

    status.info("✅ Extracting action items…")
    action_items = extract_action_items(transcript)
    progress.progress(75, text="Action items extracted")

    status.info("🔑 Extracting key decisions…")
    decisions = extract_key_decisions(transcript)
    progress.progress(85, text="Decisions extracted")

    status.info("❓ Extracting open questions…")
    questions = extract_questions(transcript)
    progress.progress(92, text="Questions extracted")

    status.info("💬 Building chat index (RAG)…")
    rag_chain = build_rag_chain(transcript)
    progress.progress(100, text="Done!")

    status.empty()
    progress.empty()

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# --------------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎬 AI Video Assistant")
    st.caption("Transcribe, summarize, and chat with any video or audio.")

    st.divider()

    input_mode = st.radio("Source type", ["YouTube URL", "Upload file"], horizontal=False)

    source = None
    uploaded_file = None

    if input_mode == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio/video file",
            type=["mp4", "mp3", "wav", "m4a", "mov", "mkv", "webm"],
        )
        if uploaded_file is not None:
            import tempfile, os
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.flush()
            source = tmp.name

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        run_clicked = st.button("🚀 Run", use_container_width=True, type="primary")
    with col_b:
        st.button("🔄 Reset", use_container_width=True, on_click=reset_state)

    if st.session_state.result:
        st.divider()
        st.success("Pipeline complete ✅")

# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #
if run_clicked:
    if not source:
        st.sidebar.error("Please provide a YouTube URL or upload a file first.")
    else:
        reset_state()
        try:
            with st.spinner("Running pipeline… this can take a while for long videos."):
                st.session_state.result = run_pipeline_with_progress(source, language)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
result = st.session_state.result

if result is None:
    st.title("🎬 AI Video Assistant")
    st.write(
        "Enter a YouTube URL or upload a local file in the sidebar, then click "
        "**Run** to transcribe, summarize, and chat with the content."
    )
    st.info("👈 Get started from the sidebar.")
else:
    st.title(f"📌 {result['title']}")

    tab_summary, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary & Insights", "📝 Full Transcript", "💬 Chat"]
    )

    # ---- Summary tab -----------------------------------------------------
    with tab_summary:
        st.subheader("📋 Summary")
        st.write(result["summary"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("✅ Action Items")
            items = result["action_items"]
            if isinstance(items, (list, tuple)):
                for item in items:
                    st.checkbox(str(item), key=f"action_{hash(item)}")
            else:
                st.write(items)

        with col2:
            st.subheader("🔑 Key Decisions")
            decisions = result["key_decisions"]
            if isinstance(decisions, (list, tuple)):
                for d in decisions:
                    st.markdown(f"- {d}")
            else:
                st.write(decisions)

        with col3:
            st.subheader("❓ Open Questions")
            questions = result["open_questions"]
            if isinstance(questions, (list, tuple)):
                for q in questions:
                    st.markdown(f"- {q}")
            else:
                st.write(questions)

        st.divider()
        st.download_button(
            "⬇️ Download Summary Report",
            data=(
                f"Title: {result['title']}\n\n"
                f"Summary:\n{result['summary']}\n\n"
                f"Action Items:\n{result['action_items']}\n\n"
                f"Key Decisions:\n{result['key_decisions']}\n\n"
                f"Open Questions:\n{result['open_questions']}\n"
            ),
            file_name=f"{result['title'] or 'summary'}.txt",
            mime="text/plain",
        )

    # ---- Transcript tab ----------------------------------------------------
    with tab_transcript:
        st.subheader("📝 Full Transcript")
        st.text_area("Transcript", result["transcript"], height=500)
        st.download_button(
            "⬇️ Download Transcript",
            data=result["transcript"],
            file_name=f"{result['title'] or 'transcript'}.txt",
            mime="text/plain",
        )

    # ---- Chat tab -----------------------------------------------------------
    with tab_chat:
        st.subheader("💬 Chat with your meeting")

        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(message)

        user_question = st.chat_input("Ask something about this video…")
        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(result["rag_chain"], user_question)
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"
                    st.write(answer)

            st.session_state.chat_history.append(("assistant", answer))