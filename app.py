import os
import base64
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

# --- Configuration ---
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_NEWS_TYPES = {"image", "video", "text", "url"}

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=genai.types.GenerationConfig(
        temperature=0.2,
        max_output_tokens=2048,
        response_mime_type="application/json",
    ),
)


# --- Helpers ---

def allowed_extension(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def build_prompt(news_type: str) -> str:
    """Return a news_type-specific system prompt."""

    base_rules = """
Rules:
- fake_percentage 0-20   → is_fake: false, verdict: LIKELY REAL
- fake_percentage 21-40  → is_fake: false, verdict: POSSIBLY REAL
- fake_percentage 41-60  → is_fake: false, verdict: UNCERTAIN
- fake_percentage 61-80  → is_fake: true,  verdict: POSSIBLY FAKE
- fake_percentage 81-100 → is_fake: true,  verdict: LIKELY FAKE
- red_flags: list specific suspicious elements you observed (empty list [] if none)
- trustworthy_signals: list credible elements you observed (empty list [] if none)
- Do NOT include any text outside the JSON object."""

    json_schema = """{
    "is_fake": true or false,
    "fake_percentage": <integer from 0 to 100>,
    "verdict": "<one of: LIKELY REAL, POSSIBLY REAL, UNCERTAIN, POSSIBLY FAKE, LIKELY FAKE>",
    "confidence": "<one of: LOW, MEDIUM, HIGH>",
    "summary": "<2-3 sentence plain-language summary of the finding>",
    "red_flags": ["<flag 1>", "<flag 2>", ...],
    "trustworthy_signals": ["<signal 1>", "<signal 2>", ...],
    "recommendation": "<one actionable sentence advising the reader what to do>"
}"""

    prompts = {
        "image": f"""You are an expert fact-checker and media analyst specializing in identifying fake news, misinformation, and manipulated media content.

Analyze this news image (screenshot, article, headline, or photo) thoroughly. Look for signs of digital manipulation, misleading cropping, out-of-context usage, watermarks from known unreliable sources, and inconsistencies in text overlays or metadata.

Respond ONLY with a valid JSON object in the following exact format:
{json_schema}
{base_rules}""",

        "video": f"""You are an expert fact-checker and digital forensics analyst specializing in detecting manipulated or misleading video content.

Analyze the provided video content thoroughly. Look for signs of deepfake manipulation, inconsistent lighting or shadows, unnatural facial movements, audio-visual sync issues, deceptive editing, missing context, and metadata anomalies. Also assess whether the video is being used out of its original context.

Respond ONLY with a valid JSON object in the following exact format:
{json_schema}
{base_rules}""",

        "text": f"""You are an expert fact-checker and journalist specializing in identifying misinformation, propaganda, and fake news in written content.

Analyze the provided text (news article, headline, social media post, or claim) thoroughly. Look for sensationalist language, logical fallacies, lack of credible sources, emotional manipulation, factual inaccuracies, anonymous attribution, and consistency with known facts.

Respond ONLY with a valid JSON object in the following exact format:
{json_schema}
{base_rules}""",

        "url": f"""You are an expert fact-checker and web analyst specializing in evaluating the credibility of online news sources and articles.

Analyze the news article or web page at the provided URL thoroughly. Evaluate the domain reputation, author credibility, publication date relevance, quality of cited sources, presence of correction policies, content sensationalism, and cross-referencing with reputable outlets.

Respond ONLY with a valid JSON object in the following exact format:
{json_schema}
{base_rules}""",
    }

    return prompts[news_type]


def parse_gemini_response(response) -> dict:
    """Strip any markdown fences and parse JSON from Gemini's response."""
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    return json.loads(raw_text)


# --- Analyzers per news_type ---

def analyze_image(file) -> dict:
    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    mime_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "webp": "image/webp", "gif": "image/gif",
    }
    mime_type = mime_map.get(ext, "image/jpeg")
    image_bytes = file.read()

    if not image_bytes:
        raise ValueError("Image file is empty.")

    image_part = {
        "mime_type": mime_type,
        "data": base64.b64encode(image_bytes).decode("utf-8"),
    }
    response = model.generate_content(contents=[{
        "role": "user",
        "parts": [{"inline_data": image_part}, {"text": build_prompt("image")}],
    }])
    return parse_gemini_response(response)


def analyze_video(file) -> dict:
    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    mime_map = {
        "mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo",
        "mkv": "video/x-matroska", "webm": "video/webm",
    }
    mime_type = mime_map.get(ext, "video/mp4")
    video_bytes = file.read()

    if not video_bytes:
        raise ValueError("Video file is empty.")

    video_part = {
        "mime_type": mime_type,
        "data": base64.b64encode(video_bytes).decode("utf-8"),
    }
    response = model.generate_content(contents=[{
        "role": "user",
        "parts": [{"inline_data": video_part}, {"text": build_prompt("video")}],
    }])
    return parse_gemini_response(response)


def analyze_text(text_content: str) -> dict:
    if not text_content or not text_content.strip():
        raise ValueError("Text content is empty.")

    prompt = build_prompt("text")
    full_prompt = f"{prompt}\n\nText to analyze:\n\"\"\"\n{text_content.strip()}\n\"\"\""

    response = model.generate_content(contents=[{
        "role": "user",
        "parts": [{"text": full_prompt}],
    }])
    return parse_gemini_response(response)


def analyze_url(url: str) -> dict:
    if not url or not url.strip():
        raise ValueError("URL is empty.")

    # Attempt to fetch the page content to give Gemini richer context
    fetched_content = ""
    try:
        page_response = requests.get(url.strip(), timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"
        })
        page_response.raise_for_status()
        # Pass raw text (truncated to avoid token overflow)
        fetched_content = page_response.text[:8000]
    except requests.RequestException as e:
        # If the page can't be fetched, still analyze using only the URL
        fetched_content = f"[Could not fetch page content: {e}]"

    prompt = build_prompt("url")
    full_prompt = (
        f"{prompt}\n\n"
        f"URL: {url.strip()}\n\n"
        f"Page content (truncated):\n\"\"\"\n{fetched_content}\n\"\"\""
    )

    response = model.generate_content(contents=[{
        "role": "user",
        "parts": [{"text": full_prompt}],
    }])
    return parse_gemini_response(response)


# --- Routes ---

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Fake News Detector API is running"}), 200


@app.route("/analyze", methods=["POST"])
def analyze_news():
    """
    Unified analysis endpoint.

    Required field (all types):
        news_type   : form field — one of: image | video | text | url

    Type-specific required fields:
        image  → multipart file field  : "image"   (png, jpg, jpeg, webp, gif)
        video  → multipart file field  : "video"   (mp4, mov, avi, mkv, webm)
        text   → form field            : "content" (plain text / article body)
        url    → form field            : "url"     (fully-qualified URL)
    """

    # --- Validate news_type ---
    news_type = request.form.get("news_type", "").strip().lower()

    if not news_type:
        return jsonify({
            "success": False,
            "error": "Missing required field 'news_type'. Must be one of: image, video, text, url.",
        }), 400

    if news_type not in ALLOWED_NEWS_TYPES:
        return jsonify({
            "success": False,
            "error": f"Invalid news_type '{news_type}'. Allowed values: {', '.join(sorted(ALLOWED_NEWS_TYPES))}.",
        }), 400

    # --- Dispatch to the correct analyzer ---
    try:
        if news_type == "image":
            if "image" not in request.files or request.files["image"].filename == "":
                return jsonify({"success": False, "error": "No image file provided. Use key 'image'."}), 400
            file = request.files["image"]
            if not allowed_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                return jsonify({
                    "success": False,
                    "error": f"Unsupported image type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
                }), 415
            analysis = analyze_image(file)

        elif news_type == "video":
            if "video" not in request.files or request.files["video"].filename == "":
                return jsonify({"success": False, "error": "No video file provided. Use key 'video'."}), 400
            file = request.files["video"]
            if not allowed_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS):
                return jsonify({
                    "success": False,
                    "error": f"Unsupported video type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
                }), 415
            analysis = analyze_video(file)

        elif news_type == "text":
            content = request.form.get("content", "")
            if not content.strip():
                return jsonify({"success": False, "error": "Missing or empty 'content' field for text analysis."}), 400
            analysis = analyze_text(content)

        elif news_type == "url":
            url = request.form.get("url", "")
            if not url.strip():
                return jsonify({"success": False, "error": "Missing or empty 'url' field for URL analysis."}), 400
            analysis = analyze_url(url)

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except json.JSONDecodeError:
        return jsonify({"success": False, "error": "Gemini returned malformed JSON. Please retry."}), 502
    except Exception as e:
        return jsonify({"success": False, "error": "Analysis failed.", "details": str(e)}), 500

    return jsonify({"success": True, "news_type": news_type, **analysis}), 200


# --- Error Handlers ---

@app.errorhandler(413)
def request_entity_too_large(_):
    return jsonify({"success": False, "error": "File too large. Maximum size is 10 MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error.", "details": str(e)}), 500


@app.route("/")
def hello():
    return "Fake News Detector API — visit /health to check status."


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    app.run(debug=False, host="0.0.0.0", port=5001)