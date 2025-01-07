# Ayurveda AI Chatbot

## Overview
An intelligent chatbot system designed to provide accurate and contextual information about Ayurvedic medicine and treatments. Built using LangChain, Llama 2, and modern vector database technology, this chatbot offers reliable medical information based on traditional Ayurvedic texts.

## Features
- Natural language understanding of medical queries
- Context-aware responses from Ayurvedic sources
- RAG (Retrieval Augmented Generation) implementation
- Vector similarity search for accurate information retrieval
- Web interface for easy interaction
- PDF document processing capabilities
- Semantic search functionality

## Architecture
```
Ayurveda Chatbot
├── Data Processing
│   ├── PDF Extractor
│   ├── Text Chunker
│   └── Data Cleaner
├── Vector Database
│   ├── Embedding Generator
│   └── Pinecone Integration
├── LLM Integration
│   ├── Llama 2 Model
│   └── LangChain Pipelines
└── Web Interface
    └── Flask Application
```

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB RAM (minimum)
- Pinecone API account
- Access to Llama 2 model

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ayurveda-chatbot.git
cd ayurveda-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Environment Setup
Create a `.env` file:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
LLAMA_MODEL_PATH=path/to/llama/model
```

### Model Configuration
```python
# config/model_config.yaml
llama:
  model_size: "7B"
  temperature: 0.7
  max_tokens: 512
  
embeddings:
  model: "sentence-transformers/all-mpnet-base-v2"
  dimension: 768
```

## Implementation

### 1. Data Processing
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataProcessor:
    def __init__(self):
        self.loader = PyPDFLoader()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def process_pdf(self, pdf_path):
        # Load PDF
        document = self.loader.load(pdf_path)
        
        # Split into chunks
        chunks = self.splitter.split_documents(document)
        
        return chunks
```

### 2. Vector Database Integration
```python
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        
    def store_embeddings(self, chunks):
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(chunks)
        
        # Store in Pinecone
        index = pinecone.Index("ayurveda-index")
        index.upsert(vectors=zip(range(len(embeddings)), embeddings))
```

### 3. Query Processing
```python
from langchain.chains import RetrievalQA
from llama import Llama

class QueryProcessor:
    def __init__(self):
        self.model = Llama()
        self.vector_store = VectorStore()
        
    def process_query(self, query):
        # Create embedding for query
        query_embedding = self.vector_store.embeddings.embed_query(query)
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query_embedding)
        
        # Generate response
        chain = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type="stuff",
            retriever=docs
        )
        
        response = chain.run(query)
        return response
```

### 4. Flask Web Interface
```python
from flask import Flask, request, jsonify
from chatbot import AyurvedaChatbot

app = Flask(__name__)
chatbot = AyurvedaChatbot()

@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query')
    
    response = chatbot.process_query(query)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
```

## Usage

### Starting the Application
```bash
# Start the Flask server
python app.py
```

### API Usage
```python
import requests

# Send query to chatbot
response = requests.post(
    'http://localhost:5000/query',
    json={'query': 'What is Ayurveda?'}
)

print(response.json()['response'])
```

## Customization

### Response Templates
```yaml
# templates/responses.yaml
greetings:
  - "Namaste! How can I assist you with Ayurvedic knowledge today?"
  - "Welcome! I'm here to help you understand Ayurvedic principles."

disclaimers:
  - "Please consult with a qualified Ayurvedic practitioner for medical advice."
  - "This information is for educational purposes only."
```

## Performance Optimization

### Vector Search Optimization
- Implement caching for frequent queries
- Use batch processing for multiple queries
- Optimize embedding dimensions

### Model Optimization
- Use quantization for reduced memory usage
- Implement response caching
- Optimize context window usage

## Troubleshooting

### Common Issues
1. PDF Processing Issues
```python
def verify_pdf_processing():
    try:
        processor = DataProcessor()
        chunks = processor.process_pdf("test.pdf")
        print(f"Successfully processed {len(chunks)} chunks")
    except Exception as e:
        print(f"PDF processing failed: {e}")
```

2. Embedding Generation Issues
```python
def test_embeddings():
    try:
        vector_store = VectorStore()
        test_text = "Test embedding generation"
        embedding = vector_store.embeddings.embed_query(test_text)
        print("Embedding generation successful")
    except Exception as e:
        print(f"Embedding generation failed: {e}")
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments
- LangChain team
- Llama model developers
- Pinecone team
- Hugging Face community
