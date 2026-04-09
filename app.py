import os
import base64
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from werkzeug.utils import secure_filename

# this method with load the configuratio( API KEY ) in our project.
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
 
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",           # stable flash model with vision support
    generation_config=genai.types.GenerationConfig(
        temperature=0.2,
        max_output_tokens=2048,              # was 800 — too low, JSON was cut off mid-string
        response_mime_type="application/json",  # forces raw JSON output, no markdown fences
    ),
)



def allowed_file_extention(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



# Method to analyze new image
# INPUTS: image bytes, mime type

def analyze_news_image(image_bytes: bytes, mime_type: str) -> dict:

    # Sends the image to Gemimi and returns a structured analysis, base on the prompt with the rules defined.

    prompt = """You are an expert fact-checker and media analyst specializing in identifying fake news, misinformation, and manipulated media content.

    Analyze this news image (screenshot, article, headline, or photo) thoroughly and respond ONLY with a valid JSON object in the following exact format:

    {
        "is_fake": true or false,
        "fake_percentage": <integer from 0 to 100>,
        "verdict": "<one of: LIKELY REAL, POSSIBLY REAL, UNCERTAIN, POSSIBLY FAKE, LIKELY FAKE>",
        "confidence": "<one of: LOW, MEDIUM, HIGH>",
        "summary": "<2-3 sentence plain-language summary of the finding>",
        "red_flags": ["<flag 1>", "<flag 2>", ...],
        "trustworthy_signals": ["<signal 1>", "<signal 2>", ...],
        "recommendation": "<one actionable sentence advising the reader what to do>"
    }

    Rules:
    - fake_percentage 0-20   → is_fake: false, verdict: LIKELY REAL
    - fake_percentage 21-40  → is_fake: false, verdict: POSSIBLY REAL
    - fake_percentage 41-60  → is_fake: false, verdict: UNCERTAIN
    - fake_percentage 61-80  → is_fake: true,  verdict: POSSIBLY FAKE
    - fake_percentage 81-100 → is_fake: true,  verdict: LIKELY FAKE
    - red_flags: list specific suspicious elements you observed (empty list [] if none)
    - trustworthy_signals: list credible elements you observed (empty list [] if none)
    - Do NOT include any text outside the JSON object."""


    # Gemini accepts raw bytes via the Part helper

    image_part = {
        "mime_type": mime_type,
        "data": base64.b64encode(image_bytes).decode("utf-8"),
    }

    response = model.generate_content(
        contents=[
            {"role": "user", "parts": [{"inline_data": image_part}, {"text": prompt}]}
        ]
    )

    raw_text = response.text.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    result = json.loads(raw_text)
    return result




# --- Routes ---

# Simple method to check if the server is up and running and enpoit working.
@app.route('/health', methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Fake News Detector API is running"}), 200






@app.route('/analyze', methods=["POST"])
def analyze_news():
    # validate request
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided. Use key 'image'."}), 400
    
    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not allowed_file_extention(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415
    
    # --- Read image bytes ---
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"success": False, "error": "Image file is empty."}), 400

    # Determine MIME type
    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/jpeg")


    # --- Call Gemini ---
    analysis = analyze_news_image(image_bytes, mime_type)

    return jsonify({"success": True, **analysis}), 200


@app.errorhandler(413)
def request_entity_too_large(_):
    return jsonify({"success": False, "error": "Image too large. Maximum size is 10 MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error.", "details": str(e)}), 500


@app.route('/')
def hello():
    return 'hello world! from new project.'


# Run the app if this file is executed
if __name__ == '__main__':
    app.run()

#    if not os.environ.get("GEMINI_API_KEY"):
#        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
#    app.run(debug=False, host="0.0.0.0", port=5001)




