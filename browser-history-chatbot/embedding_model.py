import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sockets import Sockets
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification, \
    pipeline
from urllib.parse import urlparse, parse_qs, urlunparse
import torch
import faiss
import numpy as np
import os
import json
import io
import re
import gradio as gr
import requests
import wave
import ffmpeg
import queue
import ast
import threading
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydub import AudioSegment
from google.oauth2 import service_account
from google.cloud import speech
from werkzeug.utils import secure_filename
from colorthief import ColorThief
from io import BytesIO
from PIL import Image
from collections import Counter

load_dotenv()
app = Flask(__name__)
CORS(app)
sockets = Sockets(app)

logging.basicConfig(level=logging.INFO)

AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/ffmpeg/bin/ffprobe.exe"

api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'sa-speech-history-browser.json'

title_vectorstore = None
content_vectorstore = None
history_data = []

# Load the tokenizer and model for semantic similarity
semantic_model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
semantic_model = AutoModel.from_pretrained(semantic_model_name)


# Methods for color extraction
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def parse_color(color):
    if not color or not isinstance(color, str):
        return None
    color = color.strip()
    if not color:
        return None

    try:
        if color.startswith('#'):
            hex_color = color.lstrip('#')
            if len(hex_color) == 3:  # Convert #RGB to #RRGGBB
                hex_color = ''.join(c + c for c in hex_color)
            if len(hex_color) != 6:  # Handle invalid hex string length
                raise ValueError(f"Invalid hex color length: {hex_color}")
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        elif color.startswith('rgb'):
            values = re.findall(r'\d+', color)
            if len(values) == 3:
                return tuple(map(int, values))
        else:
            color_map = {
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255),
            }
            return color_map.get(color.lower())
    except (ValueError, IndexError) as e:
        logging.warning(f"Error parsing color '{color}': {str(e)}")
        return None
    return None


def pool_colors(colors, threshold=20):
    pooled = []
    for color in colors:
        for i, pooled_color in enumerate(pooled):
            if all(abs(c1 - c2) <= threshold for c1, c2 in zip(color, pooled_color)):
                pooled[i] = tuple((c1 + c2) // 2 for c1, c2 in zip(color, pooled_color))
                break
        else:
            pooled.append(color)
    return pooled


def get_css_colors(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        inline_styles = [tag.get('style', '') for tag in soup.find_all(style=True)]
        stylesheets = [link.get('href', '') for link in soup.find_all('link', rel='stylesheet')]

        colors = []

        for style in inline_styles:
            if style:
                colors.extend(re.findall(r'#[0-9a-fA-F]{3,6}|rgb$$\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$$', style))

        for stylesheet in stylesheets:
            if not stylesheet:
                continue
            if not stylesheet.startswith(('http:', 'https:')):
                stylesheet = f"{url.rstrip('/')}/{stylesheet.lstrip('/')}"
            try:
                css_response = requests.get(stylesheet, headers=HEADERS, timeout=5)
                css_response.raise_for_status()
                colors.extend(re.findall(r'#[0-9a-fA-F]{3,6}|rgb$$\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$$', css_response.text))
            except Exception as e:
                logging.warning(f"Error fetching stylesheet {stylesheet}: {str(e)}")
                continue

        parsed_colors = []
        for color in colors:
            parsed = parse_color(color)
            if parsed:
                parsed_colors.append(parsed)

        return parsed_colors
    except Exception as e:
        logging.error(f"Error getting CSS colors from {url}: {str(e)}")
        return []


def get_image_from_url(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error fetching image from {url}: {str(e)}")
        return None


def get_dominant_color_from_image(image):
    if image is None:
        return None
    try:
        # Chuyển đổi hình ảnh sang định dạng RGB nếu cần
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Lưu hình ảnh tạm thời vào bộ nhớ
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Sử dụng ColorThief để lấy màu chủ đạo
        color_thief = ColorThief(img_byte_arr)
        return color_thief.get_color(quality=1)
    except Exception as e:
        print(f"Error getting dominant color from image: {str(e)}")
        return None


def get_dominant_color(url, default_color=(255, 255, 255)):
    try:
        # Get colors from CSS
        css_colors = get_css_colors(url)
        if css_colors:
            pooled_css_colors = pool_colors([c for c in css_colors if c is not None])
            if pooled_css_colors:
                css_color_counts = Counter(pooled_css_colors)

                # Try to get image color
                try:
                    image = get_image_from_url(url)
                    image_color = get_dominant_color_from_image(image) if image else None

                    # Combine results
                    all_colors = list(css_color_counts.keys())
                    if image_color:
                        all_colors.append(image_color)

                    pooled_all_colors = pool_colors(all_colors)
                    if pooled_all_colors:
                        return Counter(pooled_all_colors).most_common(1)[0][0]
                except Exception as e:
                    logging.warning(f"Error processing image from {url}: {str(e)}")
                    # Fall back to CSS colors only
                    return css_color_counts.most_common(1)[0][0]

        # If no CSS colors found, try image only
        image = get_image_from_url(url)
        if image:
            image_color = get_dominant_color_from_image(image)
            if image_color:
                return image_color

        logging.warning(f"No colors found for {url}, using default color")
        return default_color
    except Exception as e:
        logging.error(f"Error processing {url}: {str(e)}")
        return default_color


# Methods for website content extraction
def crawl_website_content(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = " ".join([p.get_text() for p in soup.find_all('p')])

        if not main_content:
            logging.warning("No main content found in URL: %s", url)
            return None, []

        summary = extract_summary(main_content)
        if not summary:
            logging.warning("Failed to extract summary from URL: %s", url)
            return None, []

        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', summary)
        return summary, sentences
    except requests.exceptions.RequestException as e:
        logging.error(f"Không thể truy cập URL {url}. Lỗi: {e}")
        return None, []


def extract_json_from_response(response_text):
    # Sử dụng biểu thức chính quy để trích xuất nội dung JSON từ chuỗi văn bản
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e}")
            return None
    return None


# Methods for semantic search
def normalize_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'v' in query_params:
        normalized_query = f"v={query_params['v'][0]}"
        normalized_url = urlunparse(parsed_url._replace(query=normalized_query))
        return normalized_url
    return url


def filterDuplicateHistory(historyItems):
    """
    Lọc các URL hoặc tiêu đề trùng lặp từ lịch sử duyệt web.
    """
    uniqueHistory = []
    seenUrls = set()
    seenTitles = set()

    for item in historyItems:
        normalized_url = normalize_url(item['url'])
        if normalized_url not in seenUrls and item['title'] not in seenTitles:
            uniqueHistory.append(item)
            seenUrls.add(normalized_url)
            seenTitles.add(item['title'])
    return uniqueHistory


def get_normalized_embeddings(sentences, max_length=128):
    if not sentences:
        raise ValueError("The input sentences list is empty.")
    inputs = semantic_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        embeddings = semantic_model(**inputs).pooler_output
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    return (embeddings / norms).cpu().numpy()


def semantic_search(query_embedding, vectorstore, filtered_history, k=1):
    distances, indices = vectorstore.search(query_embedding, k=k)
    results = [
        (filtered_history[i], float(distances[0][j]))
        for j, i in enumerate(indices[0])
    ]
    return results


# Methods for audio transcription
def transcribe_audio_stream(audio_generator):
    credentials = service_account.Credentials.from_service_account_file('sa-speech-history-browser.json')
    client = speech.SpeechClient(credentials=credentials)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='vi-VN'
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
    responses = client.streaming_recognize(config=streaming_config, requests=requests)

    for response in responses:
        for result in response.results:
            if result.is_final:
                logging.info(f"Transcript: {result.alternatives[0].transcript}")
                yield result.alternatives[0].transcript


# Methods for prompt-based API calls
def extract_summary(input_text):
    prompt = f"""
        Summarize the following content in vietnamese:

        {input_text}

        Output the result only.
        """

    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and "message" in result["choices"][0]:
            content = result["choices"][0]["message"]["content"]
            return content.strip()
        else:
            logging.error("Unexpected API response format.")
            return None

    except requests.exceptions.HTTPError as err:
        logging.error(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        logging.error(f"Error occurred: {err}")
    except KeyError as err:
        logging.error(f"KeyError: {err} in API response")
    except json.JSONDecodeError as err:
        logging.error(f"JSONDecodeError: {err} in API response")

    return None


def classify_website(title, content):
    prompt = f"""
        Classify the website into one or more of the following categories: entertainment, music, news, social media, education, tech, health, shopping, finance, travel, productivity, forums, sports, food, science, home.

        Title: {title}
        Content: {content}

        Output the categories as a array.

        Example:
        Title: Example Title
        Content: Example content about a tech and productivity tool.
        Output: ["tech", "productivity"]
    """

    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and "message" in result["choices"][0]:
            content = result["choices"][0]["message"]["content"]
            match = re.search(r'\[.*?\]', content)
            if match:
                categories = match.group(0)
                result = json.loads(categories)
                valid_categories = []
                for category in result:
                    if category in ["entertainment", "music", "news", "social media", "education", "tech", "health",
                                    "shopping", "finance", "travel", "productivity", "forums", "sports", "food",
                                    "science", "home"]:
                        valid_categories.append(category)
                    else:
                        logging.error("Invalid category: %s", category)
                if valid_categories:
                    return valid_categories
                else:
                    return ["other"]
        else:
            logging.error("Unexpected API response format.")
            return ["other"]

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except KeyError as err:
        print(f"KeyError: {err} in API response")
    except json.JSONDecodeError as err:
        print(f"JSONDecodeError: {err} in API response")

    return ["other"]


def extract_features(input_text):
    prompt = f"""
        Extract the following features from the text provided, returning only the output in JSON format:

        1. **time** - Look for references to times or events that may indicate when the action or request occurred, such as "yesterday at 7 PM" or "3 days ago." Normalize the time to obtain `start_date` and `end_date` in a standard format to support time constraints. 

        - For example, if the text includes "2 days ago," calculate `start_date` as two days before today’s date (today is 31/10/2024): 
            - `start_date`: 00:00 on 29/10/2024
            - `end_date`: 23:59 on 29/10/2024.
        
        - If the text specifies a time, like "today at 8 PM," convert to an hour range around the specified time:
            - `start_date`: 19:00 on 31/10/2024
            - `end_date`: 21:00 on 31/10/2024.

        2. **title** - Identify any references to topics or main themes, especially if they are phrases that start with or include "related to" or "main topic." For example, "related to food," "main theme about travel."

        3. **color** - Detect references to primary or dominant colors mentioned in the text, such as "yellow," "blue," or "green." Return the color in RGB format.

        4. **content** - Extract the main content or focus area of the text, often appearing after phrases like "main content about" or "content regarding." For example, "main content about Vietnamese bun cha dish."

        5. **category** - Classify the website into one of the following categories: entertainment, music, news, social media, education, tech, health, shopping, finance, travel, productivity, forums, sports, food, science, home.

        Example Input: "Tìm cho tôi 1 website tôi có truy cập vào 7h tối qua, web này có tiêu đề liên quan đến ẩm thực và màu chủ đạo là màu vàng, nội dung chính về món ăn bún chả Việt Nam."

        Expected JSON Output:
        {{
        "time": {{
            "original": "7h tối qua",
            "value": "2024-10-30T19:00:00",
            "start_date": "2024-10-30T18:00:00",
            "end_date": "2024-10-30T20:00:00"
        }},
        "title": "liên quan đến ẩm thực",
        "color": [255, 255, 0],
        "content": "món ăn bún chả Việt Nam",
        "category": "food"
        }}

        Text to Analyze: "{input_text}"

        Output JSON Only:
        """

    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and "message" in result["choices"][0]:
            content = result["choices"][0]["message"]["content"]
            return content
        else:
            print("Unexpected API response format.")
            return None

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except KeyError as err:
        print(f"KeyError: {err} in API response")
    except json.JSONDecodeError as err:
        print(f"JSONDecodeError: {err} in API response")

    return None


def is_question_within_scope(question):
    # Craft the prompt
    prompt = f"Is the following question within the scope of browser history search?\n\nQuestion: {question}\n\nAnswer with 'yes' or 'no'."

    # Prepare the payload for the API request
    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        # Check if the expected fields are in the response
        if "choices" in result and "message" in result["choices"][0]:
            content = result["choices"][0]["message"]["content"]
            if "Yes" in content or "yes" in content:
                return True, content
            elif "No" in content or "no" in content:
                return False, content
        else:
            print("Unexpected API response format.")
            return False, "Unexpected API response."

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        print(f"Error occurred: {err}")
    except KeyError as err:
        print(f"KeyError: {err} in API response")

    return False, "API request failed."


# Methods for calculating scores
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_time_score(query_time, page_time, max_time_diff):
    time_diff = abs(query_time - page_time)
    return 1 - (time_diff / max_time_diff)


def calculate_color_score(query_color, page_color):
    if not query_color or not page_color:
        return 0
    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(query_color, page_color)))
    max_distance = np.sqrt(3 * (255 ** 2))
    return 1 - (distance / max_distance)


def calculate_category_score(query_category, page_categories):
    return 1 if query_category in page_categories else 0


def calculate_total_score(title_score, content_score, time_score, color_score, category_score):
    return (0.4 * title_score) + (0.4 * content_score) + (0.1 * time_score) + (0.05 * color_score) + (
            0.05 * category_score)


# API Chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('query', '')
    logging.info("Received user message: %s", user_message)

    is_within_scope, relevant_title = is_question_within_scope(user_message)

    if is_within_scope:
        features_response = extract_features(user_message)
        logging.info("Extracted features response: %s", features_response)

        features = extract_json_from_response(features_response)
        if features:
            query_embedding = get_normalized_embeddings([user_message])

            # Use both title and content vectorstore
            title_results = []
            content_results = []
            color_results = []
            time_results = []

            if 'title' in features and features['title']:
                title_results = semantic_search(query_embedding, title_vectorstore, history_data)

            if 'content' in features and features['content']:
                content_results = semantic_search(query_embedding, content_vectorstore, history_data)

            if 'color' in features and features['color']:
                for item in history_data:
                    if 'color' in item:
                        color_score = calculate_color_score(features['color'], item['color'])
                        if color_score > 0:
                            color_results.append((item, color_score))

            if 'time' in features and features['time']['value']:
                query_time = datetime.strptime(features['time']['value'], "%Y-%m-%dT%H:%M:%S").timestamp()
                for item in history_data:
                    if 'lastVisitTime' in item:
                        page_time = item['lastVisitTime'] / 1000  # Convert milliseconds to seconds
                        max_time_diff = 30 * 24 * 60 * 60  # 30 days in seconds
                        time_score = calculate_time_score(query_time, page_time, max_time_diff)
                        if time_score > 0:
                            time_results.append((item, time_score))

            # Combine and deduplicate results
            combined_results = {}
            for item, score in title_results + content_results + color_results + time_results:
                if item['url'] not in combined_results:
                    combined_results[item['url']] = {'item': item, 'score': score}
                else:
                    combined_results[item['url']]['score'] = max(combined_results[item['url']]['score'], score)

            scores = []
            for url, data in combined_results.items():
                item = data['item']
                base_score = data['score']

                time_score = 0
                if 'time' in features and features['time']['value'] and features['time'][
                    'value'] != "0000-00-00T00:00:00" and features['time']['value'] != "null" and features['time'][
                    'value'] != "":
                    query_time = datetime.strptime(features['time']['value'], "%Y-%m-%dT%H:%M:%S").timestamp()
                    page_time = item['lastVisitTime'] / 1000  # Convert milliseconds to seconds
                    max_time_diff = 30 * 24 * 60 * 60  # 30 days in seconds
                    time_score = calculate_time_score(query_time, page_time, max_time_diff)

                color_score = 0
                if 'color' in features and features['color'] and 'color' in item:
                    query_color = features['color']
                    page_color = item.get('color')
                    if isinstance(query_color, list) and len(query_color) == 3:
                        color_score = calculate_color_score(query_color, page_color)
                    else:
                        parsed_color = parse_color(query_color)
                        if parsed_color:
                            color_score = calculate_color_score(parsed_color, page_color)

                category_score = 0
                if 'category' in features and 'categories' in item:
                    category_score = calculate_category_score(features['category'], item['categories'])

                total_score = calculate_total_score(base_score, base_score, time_score, color_score, category_score)
                scores.append((item, total_score))

            # Sort the results by score in descending order
            sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)

            # Return the top 3 results
            if sorted_results:
                top_3_results = sorted_results[:3]
                response = []
                for match, score in top_3_results:
                    logging.info("Match: %s, score: %f", match['title'], score)
                    response.append({
                        'url': match['url'],
                        'title': match['title'],
                        'score': score,
                        'color': match.get('color'),
                        'categories': match.get('categories', []),
                        'lastVisitTime': match['lastVisitTime']
                    })
                response = sorted(response, key=lambda x: x['score'], reverse=True)
                print('Response:', response)
                return jsonify({'response': response})
            else:
                return jsonify({'response': "Không tìm thấy tiêu đề liên quan."})
        else:
            return jsonify({'response': "Không tìm thấy tiêu đề liên quan."})
    else:
        return jsonify({'response': "Câu hỏi này không liên quan đến tiêu đề website."})


@sockets.route('/transcribe_stream')
def transcribe_socket(ws):
    audio_queue = queue.Queue()

    def audio_generator():
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def receive_audio():
        while not ws.closed:
            message = ws.receive()
            if message:
                audio_queue.put(message)
            else:
                audio_queue.put(None)
                break

    threading.Thread(target=receive_audio).start()

    for transcript in transcribe_audio_stream(audio_generator()):
        ws.send(json.dumps({"transcription": transcript}))


def save_history_to_json(new_history_data):
    file_path = 'history_data.json'
    if os.path.exists(file_path):
        # Đọc dữ liệu hiện có từ file
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Kết hợp dữ liệu hiện có với dữ liệu mới
    combined_data = existing_data + new_history_data

    # Lưu dữ liệu kết hợp trở lại file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    logging.info("History data saved to history_data.json")


@app.route('/upload_history', methods=['POST'])
def upload_history():
    """
    API xử lý embedding các mục chưa được xử lý và trả về danh sách ID đã xử lý.
    """
    global history_data, title_vectorstore, content_vectorstore

    new_history_data = request.json.get('history', [])[:10]
    logging.info("Received %d new history items", len(new_history_data))
    if not new_history_data:
        return jsonify({'status': 'error', 'message': 'Empty history data'}), 400

    # Lọc bỏ trùng lặp trước khi xử lý
    new_history_data = filterDuplicateHistory(new_history_data)

    processed_count = 0
    for item in new_history_data:
        # Kiểm tra nếu mục đã được embedded
        if item.get('is_embedded', True):
            continue

        # Crawl nội dung từ URL
        content, sentences = crawl_website_content(item['url'])
        if not content:
            item['content'] = []
            logging.warning("Failed to crawl content from URL: %s", item['url'])
        else:
            item['content'] = sentences

        # Lấy màu sắc chủ đạo từ URL
        color = get_dominant_color(item['url'])
        if not color:
            logging.warning("Failed to get dominant color from URL: %s", item['url'])
        else:
            item['color'] = color

        # Phân loại website dựa trên nội dung và tiêu đề
        categories = classify_website(item['title'], content)
        item['categories'] = categories

        # Tạo embeddings cho title và content
        title_embedding = get_normalized_embeddings([item['title']])
        content_embedding = get_normalized_embeddings([" ".join(item['content'])])

        # Thêm embeddings vào vectorstore
        if title_vectorstore is None:
            title_vectorstore = faiss.IndexFlatIP(title_embedding.shape[1])
        title_vectorstore.add(title_embedding)

        if content_vectorstore is None:
            content_vectorstore = faiss.IndexFlatIP(content_embedding.shape[1])
        content_vectorstore.add(content_embedding)

        # Đánh dấu mục này đã được embedded
        item['is_embedded'] = True
        processed_count += 1

    # Cập nhật lịch sử với các mục đã xử lý
    for item in new_history_data:
        existing_item = next((x for x in history_data if x['id'] == item['id']), None)
        if existing_item:
            existing_item.update(item)
        else:
            history_data.append(item)

    logging.info("Processed %d new history items", processed_count)
    return jsonify({'status': 'success', 'history_data': new_history_data})


if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    logging.info("Starting server on port 5000")
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
