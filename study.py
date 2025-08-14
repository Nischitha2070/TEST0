import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------------- PDF Processing -------------------------
def extract_text_from_pdf(file_path):
    """
    Extracts text from each page of the uploaded PDF.
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, chunk_size=500):
    """
    Splits the extracted text into chunks of a specified size (default 500 words).
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    return chunks


# ------------------------- Vector Store (FAISS) -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index(chunks):
    """
    Creates a FAISS index from the text chunks and their embeddings.
    """
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))  # Flat L2 distance index
    index.add(np.array(embeddings))  # Adding embeddings to the index
    return index, embeddings

def search(query, index, chunks, top_k=5):
    """
    Searches for the top_k most relevant chunks for a query using the FAISS index.
    """
    query_vector = model.encode([query])  # Encode the query
    distances, indices = index.search(np.array(query_vector), top_k)  # Search the index
    return [chunks[i] for i in indices[0]]  # Return the most relevant chunks


# ------------------------- Hugging Face LLM Answer Generator -------------------------
HUGGINGFACE_API_KEY = "your_huggingface_api_key_here"  # Replace with your actual Hugging Face API Key

# Initialize Hugging Face pipeline with a fallback model (GPT-2 as an example)
generator = pipeline("text-generation", model="gpt2", api_key=HUGGINGFACE_API_KEY)

def generate_answer(context, question):
    """
    Generates an answer based on the context using a Hugging Face model.
    """
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}"
    result = generator(prompt, max_length=300, num_return_sequences=1)
    return result[0]['generated_text']


# ------------------------- Streamlit Web App -------------------------
st.title("📚 StudyMate – AI Academic Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save the uploaded file to the 'uploads' directory
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded successfully!")

    # Process the uploaded PDF
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    if len(chunks) == 0:
        st.error("No content found in the PDF. Please upload a valid file.")
    else:
        # Create the FAISS index for searching relevant chunks
        index, _ = create_faiss_index(chunks)

        question = st.text_input("Ask a question about the uploaded material:")

        if question:
            with st.spinner('Generating answer...'):
                # Search for the most relevant chunks based on the query
                relevant_chunks = search(question, index, chunks)
                context = " ".join(relevant_chunks)  # Concatenate relevant chunks

                # Generate an answer based on the context
                answer = generate_answer(context, question)

                # Display the generated answer
                st.markdown("### ✅ Answer:")
                st.write(answer)


# ------------------------- Optional: Script Test -------------------------
if __name__ == "__main__":
    # Test code only runs when script is executed directly (not via Streamlit)
    test_text = extract_text_from_pdf("my_notes.pdf")
    test_chunks = chunk_text(test_text)
    print(f"Total Chunks: {len(test_chunks)}")
    print("First chunk preview:\n", test_chunks[0])
