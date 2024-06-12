import os
import warnings
import logging
from flask import Flask, request, jsonify, render_template
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("langchain_huggingface.llms.huggingface_endpoint").setLevel(logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="max_length is not default parameter")
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0")

app = Flask(__name__)

# Set your Huggingface token
HF_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, model_kwargs={"token": HF_TOKEN})

# Load PDF documents
loader = PyPDFDirectoryLoader("./pdfs")  # Create a folder named 'pdfs' and put your PDFs there
docs = loader.load()

# Instantiate the HuggingFaceEmbeddings object
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)

# Instantiate the RecursiveCharacterTextSplitter object
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)

# Create a Qdrant collection
qdrant_collection = Qdrant.from_documents(
    all_splits,
    embeddings,
    location=":memory:",
    collection_name="all_documents"
)

# Create a retriever
retriever = qdrant_collection.as_retriever()

# Create a RetrievalQA object
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.form.get('query')
        if not data:
            data = request.json.get('query')
        if not data:
            return jsonify({'error': 'No query provided'}), 400

        with torch.no_grad():
            result = qa.invoke(data)['result']
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Process the webhook data
    # For example, log the data
    print("Webhook received:", data)

    # Respond to the webhook
    return jsonify({'status': 'success', 'data': data}), 200

if __name__ == '__main__':
    app.run(debug=True)
