"""
Script to run all data processing steps
"""
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pickle
import re

print("=" * 60)
print("STEP 1: Loading and processing CSV")
print("=" * 60)
df = pd.read_csv('olist_full_merged.csv', low_memory=False)
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

print("\n" + "=" * 60)
print("STEP 2: Creating unique order IDs")
print("=" * 60)
df['order_customer_seller_key'] = (
    df['order_id'].astype(str) + '_' + 
    df['customer_id'].astype(str) + '_' + 
    df['seller_id'].astype(str)
)
unique_combinations = df['order_customer_seller_key'].unique()
combination_to_id = {combo: idx + 1 for idx, combo in enumerate(unique_combinations)}
df['unique_order_id'] = df['order_customer_seller_key'].map(combination_to_id)
df['order_id'] = df['unique_order_id']
df = df.drop(['order_customer_seller_key', 'unique_order_id'], axis=1)
print(f"Created {len(unique_combinations)} unique order IDs")

print("\n" + "=" * 60)
print("STEP 3: Saving processed CSV")
print("=" * 60)
df.to_csv('olist_processed.csv', index=False)
print("Saved to olist_processed.csv")

print("\n" + "=" * 60)
print("STEP 4: Creating database")
print("=" * 60)
conn = sqlite3.connect('olist_database.db')
cursor = conn.cursor()

tables = ['reviews', 'payments', 'order_items', 'products', 'sellers', 'customers', 'orders']
for table in tables:
    cursor.execute(f'DROP TABLE IF EXISTS {table}')

cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    order_status TEXT,
    order_purchase_timestamp TEXT,
    order_approved_at TEXT,
    order_delivered_customer_date TEXT,
    order_delivered_carrier_date TEXT,
    order_estimated_delivery_date TEXT,
    delivery_time_days REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT,
    customer_city TEXT,
    customer_state TEXT,
    customer_zip_code_prefix TEXT,
    customer_lat REAL,
    customer_lng REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS sellers (
    seller_id TEXT PRIMARY KEY,
    seller_city TEXT,
    seller_state TEXT,
    seller_zip_code_prefix TEXT,
    seller_lat REAL,
    seller_lng REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_category_name TEXT,
    product_category_name_english TEXT,
    product_name_lenght REAL,
    product_description_lenght REAL,
    product_photos_qty REAL,
    product_weight_g REAL,
    product_length_cm REAL,
    product_height_cm REAL,
    product_width_cm REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS order_items (
    order_id INTEGER,
    order_item_id INTEGER,
    product_id TEXT,
    seller_id TEXT,
    customer_id TEXT,
    price REAL,
    freight_value REAL,
    item_total_value REAL,
    shipping_limit_date TEXT,
    PRIMARY KEY (order_id, order_item_id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS payments (
    order_id INTEGER,
    payment_types TEXT,
    total_payment_value REAL,
    payment_count REAL,
    order_total_value REAL,
    PRIMARY KEY (order_id)
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS reviews (
    order_id INTEGER,
    avg_review_score REAL,
    review_count REAL,
    review_comments TEXT,
    review_emotion TEXT,
    review_emotion_intensity REAL,
    PRIMARY KEY (order_id)
)
''')

conn.commit()
print("Database schema created")

print("\n" + "=" * 60)
print("STEP 5: Extracting emotions from reviews")
print("=" * 60)

positive_words = [
    'perfeito', 'excelente', 'ótimo', 'bom', 'satisfeito', 'gostei', 'adorei', 'recomendo',
    'qualidade', 'rápido', 'antes do prazo', 'surpreendeu', 'muito bom', 'top', 'show',
    'maravilhoso', 'incrível', 'fantástico', 'amor', 'feliz', 'content', 'satisfeito'
]

negative_words = [
    'ruim', 'péssimo', 'horrível', 'decepcionado', 'não gostei', 'problema', 'defeito',
    'quebrado', 'danificado', 'atrasado', 'demorado', 'triste', 'chateado', 'insatisfeito',
    'não recebi', 'não veio', 'faltou', 'errado', 'diferente', 'avariação', 'reclamação'
]

def extract_emotion_from_review(review_text):
    if pd.isna(review_text) or not str(review_text).strip():
        return 'neutral', 0.0
    
    review_lower = str(review_text).lower()
    positive_count = sum(1 for word in positive_words if word in review_lower)
    negative_count = sum(1 for word in negative_words if word in review_lower)
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    emojis = emoji_pattern.findall(review_text)
    positive_emojis = ['😊', '😍', '😁', '😄', '😃', '👍', '❤️', '💯', '⭐', '🌟', '✨']
    negative_emojis = ['😢', '😞', '😠', '😡', '👎', '😥', '😔']
    
    emoji_score = 0
    for emoji in emojis:
        if any(pe in emoji for pe in positive_emojis):
            emoji_score += 1
        elif any(ne in emoji for ne in negative_emojis):
            emoji_score -= 1
    
    sentiment_score = positive_count - negative_count + (emoji_score * 0.5)
    
    if sentiment_score > 1:
        emotion = 'positive'
        intensity = min(1.0, abs(sentiment_score) / 5.0)
    elif sentiment_score < -1:
        emotion = 'negative'
        intensity = min(1.0, abs(sentiment_score) / 5.0)
    else:
        emotion = 'neutral'
        intensity = 0.5
    
    return emotion, intensity

emotions = []
intensities = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting emotions"):
    emotion, intensity = extract_emotion_from_review(row.get('review_comments', ''))
    emotions.append(emotion)
    intensities.append(intensity)

df['review_emotion'] = emotions
df['review_emotion_intensity'] = intensities
print(f"\nEmotion distribution:")
print(df['review_emotion'].value_counts())

print("\n" + "=" * 60)
print("STEP 6: Populating database")
print("=" * 60)

orders_df = df[['order_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at', 
                'order_delivered_customer_date', 'order_delivered_carrier_date', 
                'order_estimated_delivery_date', 'delivery_time_days']].drop_duplicates(subset=['order_id'])
orders_df.to_sql('orders', conn, if_exists='replace', index=False)
print(f"Inserted {len(orders_df)} orders")

customers_df = df[['customer_id', 'customer_unique_id', 'customer_city', 'customer_state', 
                   'customer_zip_code_prefix', 'customer_lat', 'customer_lng']].drop_duplicates(subset=['customer_id'])
customers_df.to_sql('customers', conn, if_exists='replace', index=False)
print(f"Inserted {len(customers_df)} customers")

sellers_df = df[['seller_id', 'seller_city', 'seller_state', 'seller_zip_code_prefix', 
                'seller_lat', 'seller_lng']].drop_duplicates(subset=['seller_id'])
sellers_df.to_sql('sellers', conn, if_exists='replace', index=False)
print(f"Inserted {len(sellers_df)} sellers")

products_df = df[['product_id', 'product_category_name', 'product_category_name_english', 
                  'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
                  'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']].drop_duplicates(subset=['product_id'])
products_df.to_sql('products', conn, if_exists='replace', index=False)
print(f"Inserted {len(products_df)} products")

order_items_df = df[['order_id', 'order_item_id', 'product_id', 'seller_id', 'customer_id',
                     'price', 'freight_value', 'item_total_value', 'shipping_limit_date']]
order_items_df.to_sql('order_items', conn, if_exists='replace', index=False)
print(f"Inserted {len(order_items_df)} order items")

payments_df = df[['order_id', 'payment_types', 'total_payment_value', 'payment_count', 
                  'order_total_value']].drop_duplicates(subset=['order_id'])
payments_df.to_sql('payments', conn, if_exists='replace', index=False)
print(f"Inserted {len(payments_df)} payments")

reviews_df = df[['order_id', 'avg_review_score', 'review_count', 'review_comments', 
                 'review_emotion', 'review_emotion_intensity']].drop_duplicates(subset=['order_id'])
reviews_df.to_sql('reviews', conn, if_exists='replace', index=False)
print(f"Inserted {len(reviews_df)} reviews")

conn.commit()
conn.close()
print("Database populated successfully!")

print("\n" + "=" * 60)
print("STEP 7: Loading processed data for embeddings")
print("=" * 60)
df = pd.read_csv('olist_processed.csv', low_memory=False)
print(f"Loaded {len(df)} rows")

print("\n" + "=" * 60)
print("STEP 8: Loading embedding model")
print("=" * 60)
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded")

print("\n" + "=" * 60)
print("STEP 9: Creating text representations")
print("=" * 60)

def create_comprehensive_text(row):
    text_parts = []
    text_parts.append(f"Order ID: {row['order_id']}")
    text_parts.append(f"Order Status: {row['order_status']}")
    text_parts.append(f"Purchase Date: {row['order_purchase_timestamp']}")
    text_parts.append(f"Approved Date: {row['order_approved_at']}")
    text_parts.append(f"Delivery Date: {row['order_delivered_customer_date']}")
    text_parts.append(f"Estimated Delivery: {row['order_estimated_delivery_date']}")
    text_parts.append(f"Delivery Time: {row['delivery_time_days']} days")
    text_parts.append(f"Customer ID: {row['customer_id']}")
    text_parts.append(f"Customer Unique ID: {row['customer_unique_id']}")
    text_parts.append(f"Customer City: {row['customer_city']}")
    text_parts.append(f"Customer State: {row['customer_state']}")
    text_parts.append(f"Customer Zip: {row['customer_zip_code_prefix']}")
    text_parts.append(f"Customer Location: ({row['customer_lat']}, {row['customer_lng']})")
    text_parts.append(f"Seller ID: {row['seller_id']}")
    text_parts.append(f"Seller City: {row['seller_city']}")
    text_parts.append(f"Seller State: {row['seller_state']}")
    text_parts.append(f"Seller Zip: {row['seller_zip_code_prefix']}")
    text_parts.append(f"Seller Location: ({row['seller_lat']}, {row['seller_lng']})")
    text_parts.append(f"Product ID: {row['product_id']}")
    text_parts.append(f"Product Category: {row['product_category_name']} ({row['product_category_name_english']})")
    text_parts.append(f"Product Name Length: {row['product_name_lenght']}")
    text_parts.append(f"Product Description Length: {row['product_description_lenght']}")
    text_parts.append(f"Product Photos: {row['product_photos_qty']}")
    text_parts.append(f"Product Weight: {row['product_weight_g']}g")
    text_parts.append(f"Product Dimensions: {row['product_length_cm']}cm x {row['product_width_cm']}cm x {row['product_height_cm']}cm")
    text_parts.append(f"Order Item ID: {row['order_item_id']}")
    text_parts.append(f"Price: R$ {row['price']}")
    text_parts.append(f"Freight Value: R$ {row['freight_value']}")
    text_parts.append(f"Item Total: R$ {row['item_total_value']}")
    text_parts.append(f"Shipping Limit: {row['shipping_limit_date']}")
    text_parts.append(f"Payment Type: {row['payment_types']}")
    text_parts.append(f"Total Payment: R$ {row['total_payment_value']}")
    text_parts.append(f"Order Total: R$ {row['order_total_value']}")
    text_parts.append(f"Payment Count: {row['payment_count']}")
    text_parts.append(f"Review Score: {row['avg_review_score']}")
    text_parts.append(f"Review Count: {row['review_count']}")
    if pd.notna(row['review_comments']) and str(row['review_comments']).strip():
        text_parts.append(f"Review Comment: {row['review_comments']}")
    if 'review_emotion' in row:
        text_parts.append(f"Review Emotion: {row['review_emotion']}")
        text_parts.append(f"Emotion Intensity: {row['review_emotion_intensity']:.2f}")
    return " | ".join([str(part) for part in text_parts if pd.notna(part)])

texts = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating texts"):
    text = create_comprehensive_text(row)
    texts.append(text)

print(f"Created {len(texts)} text representations")

print("\n" + "=" * 60)
print("STEP 10: Generating embeddings")
print("=" * 60)
batch_size = 100
embeddings = []

for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)
print(f"Generated embeddings shape: {embeddings.shape}")

print("\n" + "=" * 60)
print("STEP 11: Saving embeddings")
print("=" * 60)
np.save('embeddings.npy', embeddings)

metadata = {
    'texts': texts,
    'row_indices': list(range(len(texts)))
}

with open('embeddings_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Embeddings saved!")

print("\n" + "=" * 60)
print("STEP 12: Creating FAISS index")
print("=" * 60)
try:
    import faiss
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, 'embeddings_index.faiss')
    print("FAISS index created and saved!")
except ImportError:
    print("FAISS not available, will use numpy search")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)

