import streamlit as st
from datetime import date
from pathlib import Path
from src.ingesting import create_or_update_qdrant_collection
from src.chat import ask_invoice_question


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Invoice Chat",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# --------------------------------------------------
# Backend imports
# --------------------------------------------------
# Adapt these imports to your project.

# from src.chat import ask_invoice_question
# from src.ingestion import create_collection
# from src.metadata_extractor import extract_metadata


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def new_chat():
    chat_id = f"chat_{len(st.session_state.chats) + 1}"

    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
    }

    st.session_state.current_chat = chat_id
    st.session_state.messages = []


def select_chat(chat_id):
    st.session_state.current_chat = chat_id
    st.session_state.messages = st.session_state.chats[chat_id]["messages"]


def save_current_chat():
    if st.session_state.current_chat:
        st.session_state.chats[
            st.session_state.current_chat
        ]["messages"] = st.session_state.messages


def run_ingestion():
    """
    Launch your create_collection() here.

    Example:

        create_collection(
            folder="data/invoices",
            recreate=True
        )
    """

    with st.spinner("Ingesting invoices..."):
        # create_collection(...)
        create_or_update_qdrant_collection()

    st.success("Invoices ingested successfully.")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🧾 Invoice Chat")

    if st.button(
        "＋ New chat",
        use_container_width=True,
        type="primary",
    ):
        new_chat()
        st.rerun()

    st.divider()

    # --------------------------------------------------
    # Chats
    # --------------------------------------------------

    st.subheader("Chats")

    if not st.session_state.chats:
        st.caption("No conversations yet.")

    for chat_id, chat in st.session_state.chats.items():

        label = chat["title"]

        if st.button(
            label,
            key=f"chat_{chat_id}",
            use_container_width=True,
        ):
            select_chat(chat_id)
            st.rerun()

    st.divider()

    # --------------------------------------------------
    # Filters
    # --------------------------------------------------

    st.subheader("Invoice filters")

    date_filter = st.date_input(
        "Invoice date",
        value=None,
    )

    total_min, total_max = st.slider(
        "Total",
        min_value=0.0,
        max_value=100000.0,
        value=(0.0, 100000.0),
        step=100.0,
    )

    discount_min, discount_max = st.slider(
        "Discount",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
        format="%.0f%%",
    )

    filters = {
        "date": date_filter,
        "total_min": total_min,
        "total_max": total_max,
        "discount_min": discount_min,
        "discount_max": discount_max,
    }

    st.divider()

    # --------------------------------------------------
    # Ingestion
    # --------------------------------------------------

    st.subheader("Ingestion")

    invoice_folder = st.text_input(
        "Invoice folder",
        value="data/invoices",
    )

    st.session_state.auto_refresh = st.toggle(
        "Auto refresh",
        value=st.session_state.auto_refresh,
    )

    if st.session_state.auto_refresh:
        st.caption(
            "New invoices will be detected automatically."
        )

    if st.button(
        "⟳ Re-ingest invoices",
        use_container_width=True,
    ):
        run_ingestion()

    st.caption("Last ingestion: —")


# --------------------------------------------------
# Main area
# --------------------------------------------------

st.title("Invoice Chat")

if st.session_state.current_chat is None:

    st.markdown(
        """
        ### Ask questions about your invoices

        Search your invoices using natural language.

        **Examples**

        - Show me all invoices from September.
        - Which invoice has the highest total?
        - What was the total discount last month?
        - Find invoices above €10,000.
        - Show invoices with a discount greater than 10%.
        """
    )

else:

    # --------------------------------------------------
    # Active filters
    # --------------------------------------------------

    active_filters = []

    if date_filter:
        active_filters.append(
            f"Date: {date_filter}"
        )

    if total_min > 0:
        active_filters.append(
            f"Total ≥ {total_min:,.0f}"
        )

    if total_max < 100000:
        active_filters.append(
            f"Total ≤ {total_max:,.0f}"
        )

    if discount_min > 0:
        active_filters.append(
            f"Discount ≥ {discount_min:.0f}%"
        )

    if discount_max < 100:
        active_filters.append(
            f"Discount ≤ {discount_max:.0f}%"
        )

    if active_filters:
        st.caption(
            " · ".join(active_filters)
        )

    # --------------------------------------------------
    # Messages
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --------------------------------------------------
    # Chat input
    # --------------------------------------------------

    prompt = st.chat_input(
        "Ask something about your invoices..."
    )

    if prompt:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # --------------------------------------------------
        # Build query
        # --------------------------------------------------

        query = {
            "question": prompt,
            "filters": filters,
        }

        # --------------------------------------------------
        # Call your RAG backend
        # --------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Searching invoices..."):

                # Replace this with your existing RAG function.
                #
                # answer = ask_invoice_question(
                #     question=prompt,
                #     filters=filters,
                # )

                answer = (
                    "Connect `ask_invoice_question()` here "
                    "to your invoice RAG backend."
                )

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        save_current_chat()