import os
import tempfile

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Multi-PDF RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")
st.write(
    "Upload multiple PDF documents and ask questions across all of them."
)


# ---------------------------------------------------------
# API KEY
# ---------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not set. Please set your API key "
        "as an environment variable and restart Streamlit."
    )
    st.stop()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

st.subheader("1. Upload PDF documents")

uploaded_files = st.file_uploader(
    "Select multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ---------------------------------------------------------
# BUILD VECTOR DATABASE
# ---------------------------------------------------------

if st.button("🔨 Build Vector Database"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    all_documents = []

    progress = st.progress(0)

    for index, uploaded_file in enumerate(uploaded_files):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Add source filename to metadata
        for doc in documents:
            doc.metadata["file_name"] = uploaded_file.name

            # PyPDFLoader normally provides zero-based page metadata.
            # Convert it to human-readable page numbering.
            if "page" in doc.metadata:
                doc.metadata["page_number"] = (
                    int(doc.metadata["page"]) + 1
                )

        all_documents.extend(documents)

        os.remove(temp_path)

        progress.progress(
            (index + 1) / len(uploaded_files)
        )

    # -----------------------------------------------------
    # CHUNKING
    # -----------------------------------------------------

    st.info("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(all_documents)

    # -----------------------------------------------------
    # EMBEDDINGS
    # -----------------------------------------------------

    st.info("Creating embeddings...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # -----------------------------------------------------
    # SINGLE VECTOR DATABASE
    # -----------------------------------------------------

    st.info("Building vector database...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Store vector DB in Streamlit session state
    st.session_state.vector_store = vector_store
    st.session_state.indexed_chunks = len(chunks)
    st.session_state.pdf_names = [
        pdf.name for pdf in uploaded_files
    ]

    st.success("Vector database built successfully!")

    st.metric(
        "Chunks indexed",
        st.session_state.indexed_chunks
    )

    st.write("**PDF files indexed:**")

    for name in st.session_state.pdf_names:
        st.write(f"📄 {name}")


# ---------------------------------------------------------
# DISPLAY CURRENT VECTOR STORE STATUS
# ---------------------------------------------------------

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader("2. Ask a question")

    st.success(
        f"Ready! {st.session_state.indexed_chunks} chunks "
        f"from {len(st.session_state.pdf_names)} PDF(s) are indexed."
    )

    question = st.text_input(
        "Enter your question",
        placeholder="What does the document say about...?"
    )

    if st.button("🔍 Search and Answer"):

        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        vector_store = st.session_state.vector_store

        # -------------------------------------------------
        # RETRIEVE FROM ALL DOCUMENTS
        # -------------------------------------------------

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            st.warning(
                "No relevant information was found in the documents."
            )
            st.stop()

        # -------------------------------------------------
        # BUILD CONTEXT
        # -------------------------------------------------

        context_parts = []

        for i, doc in enumerate(retrieved_docs):

            file_name = doc.metadata.get(
                "file_name",
                "Unknown file"
            )

            page_number = doc.metadata.get(
                "page_number",
                "Unknown page"
            )

            context_parts.append(
                f"""
SOURCE {i + 1}
File: {file_name}
Page: {page_number}

Content:
{doc.page_content}
"""
            )

        context = "\n\n".join(context_parts)

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied
document context.

If the answer cannot be found in the supplied context,
say:

"I could not find this information in the uploaded documents."

Do not invent facts.

Give a clear and concise answer.
"""
                ),
                (
                    "human",
                    """
DOCUMENT CONTEXT:

{context}

QUESTION:

{question}

Answer the question using the document context.
"""
                )
            ]
        )

        chain = prompt | llm

        with st.spinner("Searching documents and generating answer..."):

            response = chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        st.subheader("Answer")

        st.write(response.content)

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        st.subheader("📚 Sources")

        seen_sources = set()

        for doc in retrieved_docs:

            file_name = doc.metadata.get(
                "file_name",
                "Unknown file"
            )

            page_number = doc.metadata.get(
                "page_number",
                "Unknown page"
            )

            source_key = (
                file_name,
                page_number
            )

            if source_key not in seen_sources:

                seen_sources.add(source_key)

                st.markdown(
                    f"**📄 {file_name} — Page {page_number}**"
                )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📊 RAG Status")

    if st.session_state.vector_store is not None:

        st.metric(
            "Indexed chunks",
            st.session_state.indexed_chunks
        )

        st.metric(
            "PDF documents",
            len(st.session_state.pdf_names)
        )

        st.write("**Documents:**")

        for name in st.session_state.pdf_names:
            st.caption(f"📄 {name}")

    else:
        st.info(
            "Upload PDFs and click "
            "'Build Vector Database'."
        )
     doc.metadata["file_name"] = uploaded_file.name
doc.metadata["page_number"] = int(doc.metadata["page"]) + 1
   st.session_state.vector_store