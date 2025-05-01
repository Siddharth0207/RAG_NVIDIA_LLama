from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import streamlit as st
from dotenv import load_dotenv
import fitz
load_dotenv()


embedding_model = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        type='embeddings',
        truncate="NONE", 
)
client = ChatNVIDIA(
  model="meta/llama-3.1-8b-instruct",
  type="chat",
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
)

# Initialize session state for logs
if "logs" not in st.session_state:
    st.session_state.logs = []

st.header("NVIDIA Powered Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "".join([page.get_text() for page in doc])
        
    st.success("PDF Parsed Sucessfully!")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])

    # Embed and store in FAISS
    vectorstore = FAISS.from_documents(docs,embedding_model)

    st.session_state.vectordb = vectorstore
    st.session_state.source_chunks = docs
    st.success("Document embedded into vector DB.")

# Chat
user_query = st.text_input("Ask your question:")


if st.button("Submit") and user_query:
    if "vectordb" not in st.session_state:
        st.error("Please upload and process a document first.")
    else:
        # Retrieve relevant docs
        relevant_docs = st.session_state.vectordb.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # Strong prompt grounding
        prompt = f"""You are a helpful assistant. Use ONLY the following document excerpts to answer the question.
                If the answer is not in the text, say "I couldn't find that information in the document.
                
                Document excerpts:
                {context}
Question: {user_query}

Answer: 

                """

        response = client.invoke(user_query)
        st.write(response.content)
        st.subheader("Sources used:")
        for i, doc in enumerate(relevant_docs, 1):
            st.markdown(f"**Source {i}:** {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")