import os
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from groq import Groq

# ---------------------------------------------------
# LOAD ENV VARIABLES
# ---------------------------------------------------

load_dotenv()

# ---------------------------------------------------
# GET GROQ API KEY
# ---------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ---------------------------------------------------
# VALIDATE API KEY
# ---------------------------------------------------

if not GROQ_API_KEY:

    st.error("❌ GROQ_API_KEY not found.")

    st.stop()

# ---------------------------------------------------
# INITIALIZE GROQ CLIENT
# ---------------------------------------------------

client = Groq(
    api_key=GROQ_API_KEY
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI YouTube Course Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🎓 AI YouTube Course Assistant")

st.markdown(
    """
AI-powered RAG chatbot for:

✅ YouTube Courses  
✅ PDF Notes  
✅ Study Materials  

Ask questions and get AI-generated answers.
"""
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📂 Upload Course Materials")

st.sidebar.markdown(
    """
### Supported Inputs
- YouTube Videos
- PDF Notes
"""
)

# ---------------------------------------------------
# YOUTUBE URL INPUT
# ---------------------------------------------------

youtube_url = st.sidebar.text_input(
    "🔗 Enter YouTube Video URL"
)

# ---------------------------------------------------
# PDF UPLOADER
# ---------------------------------------------------

uploaded_pdfs = st.sidebar.file_uploader(
    "📄 Upload PDF Notes",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# LIMIT PDF UPLOADS
# ---------------------------------------------------

if uploaded_pdfs and len(uploaded_pdfs) > 5:

    st.sidebar.error(
        "⚠ Maximum 5 PDFs allowed"
    )

    st.stop()

# ---------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    return model

embedding_model = load_embedding_model()

# ---------------------------------------------------
# INITIALIZE CHAT HISTORY
# ---------------------------------------------------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# ---------------------------------------------------
# FUNCTION TO EXTRACT VIDEO ID
# ---------------------------------------------------

def extract_video_id(youtube_url):

    try:

        parsed_url = urlparse(youtube_url)

        # Short URL
        if parsed_url.hostname == "youtu.be":

            return parsed_url.path[1:]

        # Normal URL
        if parsed_url.hostname in (
            "www.youtube.com",
            "youtube.com"
        ):

            if parsed_url.path == "/watch":

                return parse_qs(
                    parsed_url.query
                )["v"][0]

    except:

        return None

# ---------------------------------------------------
# FUNCTION TO FETCH YOUTUBE TRANSCRIPT
# ---------------------------------------------------

def get_youtube_transcript(video_id):

    try:

        ytt_api = YouTubeTranscriptApi()

        transcript = ytt_api.fetch(
            video_id
        ).to_raw_data()

        full_text = " ".join(
            [item["text"] for item in transcript]
        )

        return full_text

    except Exception as e:

        st.error(
            f"❌ Error fetching transcript: {e}"
        )

        return None

# ---------------------------------------------------
# FUNCTION TO EXTRACT PDF TEXT
# ---------------------------------------------------

def extract_pdf_text(uploaded_files):

    all_text = ""

    try:

        for pdf_file in uploaded_files:

            pdf_reader = PdfReader(pdf_file)

            for page in pdf_reader.pages:

                text = page.extract_text()

                if text:

                    all_text += text + "\n"

        return all_text

    except Exception as e:

        st.error(
            f"❌ Error reading PDF: {e}"
        )

        return None

# ---------------------------------------------------
# FUNCTION TO SPLIT TEXT
# ---------------------------------------------------

def split_text_into_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,
        chunk_overlap=100,
        length_function=len

    )

    chunks = text_splitter.split_text(text)

    return chunks

# ---------------------------------------------------
# FUNCTION TO CREATE EMBEDDINGS
# ---------------------------------------------------

def create_embeddings(chunks):

    embeddings = embedding_model.encode(
        chunks
    )

    return embeddings

# ---------------------------------------------------
# FUNCTION TO CREATE FAISS INDEX
# ---------------------------------------------------

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index

# ---------------------------------------------------
# FUNCTION TO RETRIEVE CHUNKS
# ---------------------------------------------------

def retrieve_relevant_chunks(
    query,
    index,
    chunks,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [query]
    )

    distances, indices = index.search(

        np.array(query_embedding),

        top_k

    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(
            chunks[idx]
        )

    return retrieved_chunks

# ---------------------------------------------------
# FUNCTION TO GENERATE AI RESPONSE
# ---------------------------------------------------

def generate_ai_response(
    question,
    retrieved_chunks
):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are an AI Course Assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Provide a clear, concise and helpful answer.
"""

    try:

        completion = client.chat.completions.create(

            model="llama3-8b-8192",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3

        )

        response = completion.choices[0].message.content

        return response

    except Exception as e:

        return f"❌ Error generating response: {e}"

# ---------------------------------------------------
# PROCESS CONTENT BUTTON
# ---------------------------------------------------

if st.sidebar.button("📥 Process Content"):

    all_text = ""

    # ---------------------------------------------------
    # PROCESS YOUTUBE
    # ---------------------------------------------------

    if youtube_url:

        with st.spinner(
            "Fetching YouTube transcript..."
        ):

            video_id = extract_video_id(
                youtube_url
            )

            if video_id:

                transcript = get_youtube_transcript(
                    video_id
                )

                if transcript:

                    st.success(
                        "✅ Transcript fetched successfully!"
                    )

                    all_text += transcript

            else:

                st.error(
                    "❌ Invalid YouTube URL"
                )

    # ---------------------------------------------------
    # PROCESS PDFS
    # ---------------------------------------------------

    if uploaded_pdfs:

        with st.spinner(
            "Reading PDF files..."
        ):

            pdf_text = extract_pdf_text(
                uploaded_pdfs
            )

            if pdf_text:

                st.success(
                    "✅ PDF text extracted successfully!"
                )

                all_text += pdf_text

    # ---------------------------------------------------
    # CREATE CHUNKS
    # ---------------------------------------------------

    if all_text:

        with st.spinner(
            "Creating chunks..."
        ):

            chunks = split_text_into_chunks(
                all_text
            )

            st.success(
                f"✅ Created {len(chunks)} chunks"
            )

        # ---------------------------------------------------
        # CREATE EMBEDDINGS
        # ---------------------------------------------------

        with st.spinner(
            "Creating embeddings..."
        ):

            embeddings = create_embeddings(
                chunks
            )

            st.success(
                "✅ Embeddings created"
            )

        # ---------------------------------------------------
        # CREATE FAISS INDEX
        # ---------------------------------------------------

        with st.spinner(
            "Creating vector database..."
        ):

            faiss_index = create_faiss_index(
                embeddings
            )

            st.success(
                "✅ FAISS database ready"
            )

        # ---------------------------------------------------
        # SAVE IN SESSION
        # ---------------------------------------------------

        st.session_state["chunks"] = chunks

        st.session_state["faiss_index"] = faiss_index

        st.success(
            "🎉 AI Course Assistant Ready!"
        )

    else:

        st.warning(
            "⚠ Please upload PDFs or enter YouTube URL"
        )

# ---------------------------------------------------
# CHATBOT SECTION
# ---------------------------------------------------

st.subheader("💬 Ask Questions")

user_question = st.chat_input(
    "Ask a question about the course..."
)

# ---------------------------------------------------
# HANDLE USER QUESTION
# ---------------------------------------------------

if user_question:

    st.chat_message("user").write(
        user_question
    )

    if "faiss_index" in st.session_state:

        with st.spinner(
            "Searching course content..."
        ):

            retrieved_chunks = retrieve_relevant_chunks(

                query=user_question,

                index=st.session_state["faiss_index"],

                chunks=st.session_state["chunks"],

                top_k=3

            )

        with st.spinner(
            "Generating AI response..."
        ):

            ai_response = generate_ai_response(

                user_question,

                retrieved_chunks

            )

        # ---------------------------------------------------
        # SAVE CHAT HISTORY
        # ---------------------------------------------------

        st.session_state.chat_history.append(
            {
                "question": user_question,
                "answer": ai_response
            }
        )

        # ---------------------------------------------------
        # SHOW RESPONSE
        # ---------------------------------------------------

        st.chat_message("assistant").write(
            ai_response
        )

        # ---------------------------------------------------
        # SHOW RETRIEVED CHUNKS
        # ---------------------------------------------------

        with st.expander(
            "📚 Retrieved Context"
        ):

            for i, chunk in enumerate(
                retrieved_chunks
            ):

                st.write(
                    f"### Chunk {i+1}"
                )

                st.write(
                    chunk[:700]
                )

                st.write("---")

    else:

        st.warning(
            "⚠ Please process content first"
        )

# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------

if st.session_state.chat_history:

    st.subheader("🕘 Chat History")

    for chat in reversed(
        st.session_state.chat_history
    ):

        with st.container():

            st.markdown(
                f"""
### 👤 Question
{chat['question']}

### 🤖 Answer
{chat['answer']}
"""
            )

            st.write("---")