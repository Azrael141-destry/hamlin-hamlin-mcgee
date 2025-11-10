"""
RAG-based Assistant for Olist Data
Uses OpenRouter API with free LLMs
"""

import os
import sqlite3
import numpy as np
import pickle
import requests
from typing import List, Dict, Optional
import json

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, using numpy-based search")

from sentence_transformers import SentenceTransformer


class OlistRAGAssistant:
    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-3.2-3b-instruct:free"):
        """
        Initialize the RAG Assistant
        
        Args:
            api_key: OpenRouter API key
            model_name: Model to use (default: free Llama 3.2 3B)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load embeddings and metadata
        print("Loading embeddings...")
        self.embeddings = np.load('embeddings.npy')
        with open('embeddings_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            self.texts = metadata['texts']
        
        # Load FAISS index if available, otherwise use numpy
        if FAISS_AVAILABLE:
            try:
                self.index = faiss.read_index('embeddings_index.faiss')
                print("FAISS index loaded")
            except:
                self.index = None
                print("FAISS index not found, using numpy search")
        else:
            self.index = None
        
        # Normalize embeddings for cosine similarity
        self.embeddings_normalized = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Load database
        self.db_conn = sqlite3.connect('olist_database.db')
        print("RAG Assistant initialized!")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar contexts using vector similarity"""
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
        
        if self.index is not None and FAISS_AVAILABLE:
            # Use FAISS for fast search
            query_embedding_normalized = query_embedding_normalized.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query_embedding_normalized, top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                results.append({
                    'text': self.texts[idx],
                    'score': float(1 - distances[0][i]),  # Convert distance to similarity
                    'index': int(idx)
                })
        else:
            # Use numpy for cosine similarity
            similarities = np.dot(self.embeddings_normalized, query_embedding_normalized)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'text': self.texts[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return results
    
    def get_context_from_db(self, order_id: Optional[int] = None) -> str:
        """Get additional context from database if order_id is provided"""
        if order_id is None:
            return ""
        
        cursor = self.db_conn.cursor()
        
        # Get comprehensive order information
        query = """
        SELECT 
            o.order_id, o.order_status, o.order_purchase_timestamp,
            c.customer_city, c.customer_state,
            s.seller_city, s.seller_state,
            p.product_category_name_english,
            pay.payment_types, pay.total_payment_value,
            r.avg_review_score, r.review_comments
        FROM orders o
        LEFT JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN customers c ON oi.customer_id = c.customer_id
        LEFT JOIN sellers s ON oi.seller_id = s.seller_id
        LEFT JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN payments pay ON o.order_id = pay.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_id = ?
        LIMIT 1
        """
        
        cursor.execute(query, (order_id,))
        result = cursor.fetchone()
        
        if result:
            return f"Order {result[0]}: Status={result[1]}, Customer={result[3]}, {result[4]}, Seller={result[5]}, {result[6]}, Category={result[7]}, Payment={result[8]}, Value={result[9]}, Review={result[10]}"
        return ""
    
    def query_llm(self, user_query: str, context_texts: List[str]) -> str:
        """Query the LLM via OpenRouter API"""
        
        # Build context - limit length to avoid token limits
        context_parts = []
        total_length = 0
        max_context_length = 8000  # Limit context to avoid token issues
        
        for i, text in enumerate(context_texts):
            if total_length + len(text) > max_context_length:
                break
            context_parts.append(f"Context {i+1}: {text}")
            total_length += len(text)
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        system_prompt = """You are a helpful assistant specialized in analyzing Olist e-commerce data. 
You have access to comprehensive order, customer, seller, product, payment, and review information.
Your responses should be:
- Simple and easy to understand
- Clear and concise
- Use plain language, avoid jargon
- Break down complex information into simple points
- Use bullet points or short paragraphs when helpful
- Focus on the key information the user asked for"""

        user_prompt = f"""Based on the following context from the Olist database, answer this question in a simple and easy-to-understand way: {user_query}

Context:
{context}

Please provide a clear, simple answer based on the context. Use plain language and make it easy to understand."""

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5001",
            "X-Title": "Olist RAG Assistant"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
            
            # Better error handling
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get('error', {}).get('message', error_detail)
                except:
                    pass
                return f"Error querying LLM (Status {response.status_code}): {error_detail}"
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return f"Unexpected response format: {result}"
                
        except requests.exceptions.RequestException as e:
            return f"Error querying LLM: {str(e)}"
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def ask(self, question: str, top_k: int = 5) -> Dict:
        """Main method to ask a question"""
        # Search for similar contexts
        similar_results = self.search_similar(question, top_k=top_k)
        
        # Extract context texts
        context_texts = [result['text'] for result in similar_results]
        
        # Query LLM
        answer = self.query_llm(question, context_texts)
        
        return {
            'question': question,
            'answer': answer,
            'contexts_used': len(context_texts),
            'similarity_scores': [r['score'] for r in similar_results]
        }
    
    def close(self):
        """Close database connection"""
        self.db_conn.close()


# Free LLM models available on OpenRouter
FREE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-2b-it:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "qwen/qwen-2-1.5b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "huggingface/zephyr-7b-beta:free"
]

