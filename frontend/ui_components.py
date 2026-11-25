# rag_agent_app/frontend/ui_components.py

import streamlit as st
from backend_api import upload_document_to_backend, chat_with_backend_agent
from session_manager import init_session_state # Import to access session state

def display_header():
    """Renders the main title and introductory markdown."""
    st.set_page_config(page_title="AI Agent Chatbot", layout="wide") # Set page config here
    st.title("🤖 AI Agent Chatbot")
    st.markdown("Ask me about the documents you have uploaded. I answer **only** from the knowledge base; if it is not there, I will say so. Messages are screened by input safety guardrails.")
    st.markdown("---")

def render_document_upload_section(fastapi_base_url: str):
    """
    Renders the UI for uploading PDF documents to the knowledge base.
    Handles file upload and API call to the backend.
    """
    st.header("Upload Document to Knowledge Base")
    with st.expander("Upload New Document (PDF Only)"):
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
        if st.button("Upload PDF", key="upload_pdf_button"):
            if uploaded_file is not None:
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    try:
                        upload_data = upload_document_to_backend(fastapi_base_url, uploaded_file)
                        st.success(f"PDF '{upload_data.get('filename')}' uploaded successfully! Indexed {upload_data.get('processed_chunks')} chunks.")
                    except Exception as e:
                        st.error(f"An error occurred during upload: {e}")
            else:
                st.warning("Please upload a PDF file before clicking 'Upload PDF'.")
    st.markdown("---")

def display_chat_history():
    """Displays all messages currently in the session state chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_trace_events(trace_events: list):
    """
    Renders the detailed agent workflow trace in an expandable section.
    Uses icons and conditional styling for better readability.
    """
    if trace_events:
        with st.expander("🔬 Agent Workflow Trace"):
            for event in trace_events:
                icon_map = {
                    'guardrail': "🛡️",
                    'router': "➡️",
                    'rag_lookup': "📚",
                    'not_found': "📭",
                    'answer': "💡",
                    '__end__': "✅"
                }
                icon = icon_map.get(event['node_name'], "⚙️")
                
                st.subheader(f"{icon} Step {event['step']}: {event['node_name']}")
                st.write(f"**Description:** {event['description']}")
                
                details = event['details']
                if event['node_name'] == 'guardrail':
                    if event['event_type'] == 'guardrail_block':
                        st.error(f"**Blocked ({details.get('category')}, layer: {details.get('layer')}).** {details.get('reason', '')}")
                    elif details.get('pii_found'):
                        st.warning(f"PII redacted before processing: {', '.join(details['pii_found'])}")
                    else:
                        st.success("Passed input safety checks.")
                elif event['node_name'] == 'rag_lookup' and 'sufficiency_verdict' in details:
                    verdict = details['sufficiency_verdict']
                    if verdict == "Sufficient":
                        st.success(f"**RAG Verdict:** {verdict} - relevant info found in the knowledge base.")
                    else:
                        st.warning(f"**RAG Verdict:** {verdict} - the knowledge base does not contain this information.")
                    if details.get('retrieved_content_summary'):
                        st.markdown(f"**Top chunk:** `{details['retrieved_content_summary']}`")
                elif details:
                    st.json(details)

                st.markdown("---")