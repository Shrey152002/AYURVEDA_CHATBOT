from flask import Flask, render_template, request, jsonify
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import os
import time
from pinecone import Pinecone, ServerlessSpec, Index
from tqdm.auto import tqdm

app = Flask(__name__)

# Initialize Pinecone with API key and serverless spec
API_KEY = "xxxxxxxxx" # Replace with your actual API key
INDEX_NAME = "ayurveda"  # Replace with your actual index name
DATA_DIR = "C://Users//Shreyash Verma//AYURVEDA_CHATBOT//data"  # Directory containing PDFs
MODEL_PATH = "C://Users//Shreyash Verma//AYURVEDA_CHATBOT//model//llama-2-7b-chat.ggmlv3.q4_0.bin"  # Replace with your model path

# Initialize Pinecone
pc = Pinecone(api_key=API_KEY)

# Create or get the index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        INDEX_NAME,
        dimension=384,  # Set this to the dimension of your embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

# Load PDF data
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

# Split text into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)

# Download embeddings
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Upsert data into Pinecone
def upsert_data(text_chunks):
    embeddings = download_hugging_face_embeddings()
    
    for i in tqdm(range(0, len(text_chunks), 100)):
        i_end = min(len(text_chunks), i + 100)
        batch = text_chunks[i:i_end]
        
        ids = [f"{x.metadata['source']}-{x.metadata['page']}-{i}" for i, x in enumerate(batch)]
        texts = [x.page_content for x in batch]
        embeds = embeddings.embed_documents(texts)
        
        metadata = [{'text': x.page_content, 'source': x.metadata['source'], 'page': x.metadata['page']} for x in batch]
        index.upsert(vectors=zip(ids, embeds, metadata))

# Load and upsert data if not done
if index.describe_index_stats()['total_vector_count'] == 0:
    extracted_data = load_pdf(DATA_DIR)
    text_chunks = text_split(extracted_data)
    upsert_data(text_chunks)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get("query_text")
    
    # Embed the query text to get the query vector
    embeddings = download_hugging_face_embeddings()
    query_vector = embeddings.embed_query(user_query)

    # Perform the similarity search using Pinecone's query method
    response = index.query(
        vector=query_vector,
        top_k=2,
        include_values=True,
        include_metadata=True
    )

    # Collect the top results
    similar_texts = [match['metadata']['text'] for match in response['matches']]
    
    # Generate an augmented prompt
    augmented_prompt = f"""
    Context:
    {similar_texts[0]}

    {similar_texts[1]}

    Based on the above context, provide an informative and concise answer to the following question and if you don't know say 'sorry':
    Question: {user_query}
    Answer:
    """

    # Load the language model
    llm = CTransformers(model=MODEL_PATH, model_type="llama", config={'max_new_tokens': 512, 'temperature': 0.8})

    # Generate a response using the language model
    output = llm(augmented_prompt)

    return jsonify({"response": output})

if __name__ == "__main__":
    app.run(debug=True)
