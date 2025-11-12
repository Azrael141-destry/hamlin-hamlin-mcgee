# Olist RAG Assistant

A comprehensive RAG (Retrieval-Augmented Generation) application for analyzing Olist e-commerce data. This system processes order, customer, seller, product, payment, and review data to provide intelligent answers using free LLMs via OpenRouter API.("here are the files for this application to run "https://drive.google.com/drive/folders/1GSEHhQErf5YyMmnPwZFcPHWhAPB-eDcp?usp=drive_link"")

## Features

- ✅ **Complete Data Coverage**: Processes all columns from the merged CSV
- ✅ **Unique Order IDs**: Creates unique order IDs based on order_id + customer_id + seller_id combinations
- ✅ **Relational Database**: Structured SQLite database with proper relationships
- ✅ **Vector Embeddings**: Generates embeddings for all data columns using Sentence Transformers
- ✅ **RAG System**: Retrieval-Augmented Generation with semantic search
- ✅ **Free LLMs**: Uses free models from OpenRouter API
- ✅ **Web Interface**: Beautiful, modern web UI for querying the data

- <img width="1293" height="902" alt="image" src="https://github.com/user-attachments/assets/d182da6b-db00-4ce7-9d47-8087f640e0aa" />

<img width="1542" height="848" alt="image" src="https://github.com/user-attachments/assets/2f93367a-6aec-494f-904d-a10a75eb4f3a" />


## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Process Data and Generate Embeddings

Open and run the Jupyter notebook `data_processing.ipynb`:

```bash
jupyter notebook data_processing.ipynb
```

This will:
- Process the CSV file and create unique order IDs
- Create a relational SQLite database (`olist_database.db`)
- Generate vector embeddings (`embeddings.npy`, `embeddings_metadata.pkl`)
- Create FAISS index for fast similarity search (`embeddings_index.faiss`)

### 3. Get OpenRouter API Key

1. Go to [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. The system uses free models by default

## Running the Application

### Start the Flask Web Server

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Usage

1. Open your browser and go to `http://localhost:5000`
2. Enter your OpenRouter API key
3. Select a model (free models are pre-selected)
4. Click "Initialize Assistant"
5. Start asking questions!

## Example Questions

- "What are the top product categories by sales?"
- "What is the average delivery time?"
- "Which states have the most orders?"
- "What is the average review score?"
- "What payment methods are most common?"
- "Show me orders from São Paulo"
- "What products have the highest review scores?"
- "Which customers have placed the most orders?"

## Data Structure

The system processes and indexes:

- **Order Details**: Status, purchase dates, delivery dates, estimated dates, approved dates
- **Customer Details**: IDs, cities, states, zip codes, geolocation
- **Seller Details**: IDs, cities, states, zip codes, geolocation
- **Product Details**: Categories, names, descriptions, dimensions, weights, photos
- **Payment Details**: Types, values, installments, sequential
- **Review Details**: Scores, comment titles, messages, creation dates

## Files Generated

- `olist_processed.csv`: Processed CSV with unique order IDs
- `olist_database.db`: SQLite relational database
- `embeddings.npy`: Vector embeddings array
- `embeddings_metadata.pkl`: Metadata for embeddings
- `embeddings_index.faiss`: FAISS index for fast search (optional)

## Architecture

1. **Data Processing** (`data_processing.ipynb`):
   - Loads and processes CSV
   - Creates unique order IDs
   - Builds relational database
   - Generates vector embeddings

2. **RAG System** (`rag_app.py`):
   - Semantic search using vector embeddings
   - Context retrieval from database
   - LLM integration via OpenRouter API

3. **Web Interface** (`app.py` + `templates/index.html`):
   - Flask backend API
   - Modern, responsive frontend
   - Real-time chat interface

## Free LLM Models Available

- `meta-llama/llama-3.2-3b-instruct:free`
- `google/gemma-2-2b-it:free`
- `microsoft/phi-3-mini-128k-instruct:free`
- `qwen/qwen-2-1.5b-instruct:free`

## Notes

- The first run (processing data and generating embeddings) may take some time depending on your system
- FAISS is optional but recommended for faster search on large datasets
- The system uses cosine similarity for semantic search
- All data is processed locally; only queries are sent to OpenRouter API

## Troubleshooting

- **FAISS not available**: The system will fall back to numpy-based search (slower but works)
- **Memory issues**: Reduce batch size in the notebook if you encounter memory errors
- **API errors**: Check your OpenRouter API key and account status

