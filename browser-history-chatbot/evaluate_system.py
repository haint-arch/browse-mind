import json
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from embedding_model import crawl_website_content, semantic_search, get_normalized_embeddings, title_vectorstore, content_vectorstore, history_data, calculate_total_score, calculate_time_score, calculate_color_score, calculate_category_score

# Hàm tích hợp tìm kiếm từ backend

