"""
Smart Sticker Maker MCP Server
- Flask app exposing MCP-compatible endpoints:
  - POST /validate           -> validate bearer token, return owner's phone
  - POST /generate_sticker   -> accept image (base64 or URL) + optional mood
       -> remove background, detect mood (if not provided) via Gemini
       -> generate caption via Gemini matching mood
       -> render caption onto sticker (512x512 PNG, transparent)
       -> return sticker_url, caption, detected_mood

Environment variables required:
- GEMINI_API_KEY  -> your Gemini API key
- OWNER_PHONE     -> owner's phone number in format {country_code}{number}, e.g. 919876543210
- VALIDATE_TOKEN  -> a token string that MCP clients must pass to /validate (for hackathon you can hardcode/test)

Deploy notes:
- Ensure the server is publicly accessible over HTTPS (Vercel/Render/Cloudflare)
- Serve the `static/` directory so returned sticker URLs are reachable

"""

import os
import io
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

# Optional: rembg for local background removal. If not available, we fallback to keeping background.
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

app = Flask(__name__, static_folder="static")

# Load configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_PHONE = os.getenv("OWNER_PHONE", "919440473900")
VALIDATE_TOKEN = os.getenv("VALIDATE_TOKEN", "stickypuch")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Set GEMINI_API_KEY environment variable before deploying.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Ensure static directory
os.makedirs(app.static_folder, exist_ok=True)

# Helpers

def _save_image(img: Image.Image, filename: str) -> str:
    path = os.path.join(app.static_folder, filename)
    img.save(path, format="PNG")
    return path


def _download_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")


def _decode_base64_image(b64str: str) -> Image.Image:
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert("RGBA")


def _remove_background(img: Image.Image) -> Image.Image:
    if REMBG_AVAILABLE:
        try:
            # rembg expects bytes-like image; operate with PIL Image
            output = remove(img)
            return output.convert("RGBA")
        except Exception as e:
            print("rembg failed:", e)
            return img
    else:
        # If rembg not available, return original image
        return img


def _resize_and_center(img: Image.Image, size=(512, 512)) -> Image.Image:
    # Create transparent background
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    # Fit image preserving aspect ratio
    img.thumbnail((size[0] - 20, size[1] - 100), Image.LANCZOS)
    x = (size[0] - img.width) // 2
    y = max(10, (size[1] - img.height) // 2)
    canvas.paste(img, (x, y), img)
    return canvas


def _overlay_caption(img: Image.Image, caption: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Load a truetype font if available, else default
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 32)
    except Exception:
        font = ImageFont.load_default()

    # Wrap text to fit width
    max_width = width - 40
    lines = []
    words = caption.split()
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        if draw.textsize(test, font=font)[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    # Calculate y for text block (bottom area)
    total_text_height = sum(draw.textsize(line, font=font)[1] + 6 for line in lines)
    y_text = height - total_text_height - 20

    # Draw semi-transparent rounded rectangle behind text for readability
    margin = 20
    rect_top = y_text - 10
    rect_bottom = height - 10
    rect_left = margin
    rect_right = width - margin
    # semi-transparent black
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(rect_left, rect_top), (rect_right, rect_bottom)], fill=(0, 0, 0, 120))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Draw text with white stroke for readability
    for line in lines:
        w, h = draw.textsize(line, font=font)
        x_text = (width - w) // 2
        # stroke
        draw.text((x_text-1, y_text-1), line, font=font, fill=(0,0,0,255))
        draw.text((x_text+1, y_text-1), line, font=font, fill=(0,0,0,255))
        draw.text((x_text-1, y_text+1), line, font=font, fill=(0,0,0,255))
        draw.text((x_text+1, y_text+1), line, font=font, fill=(0,0,0,255))
        # main
        draw.text((x_text, y_text), line, font=font, fill=(255,255,255,255))
        y_text += h + 6

    return img


# ------------------
# MCP-compatible endpoints
# ------------------

@app.route("/validate", methods=["POST"])
def validate():
    """Required by Puch MCP connect flow. Expects JSON: {"bearer_token": "..."}
    Should return phone number in format {country_code}{number} on success.
    """
    data = request.get_json(force=True)
    token = data.get("bearer_token")
    if not token:
        return jsonify({"error": "bearer_token required"}), 400

    # Very simple validation for hackathon: compare against VALIDATE_TOKEN
    if token == VALIDATE_TOKEN:
        return jsonify({"phone_number": OWNER_PHONE})
    else:
        return jsonify({"error": "invalid token"}), 403


@app.route("/generate_sticker", methods=["POST"])
def generate_sticker():
    """Accepts JSON with either `image_base64` or `image_url`, and optional `mood`.
    Returns: { "sticker_url": "...", "caption": "...", "mood": "..." }
    """
    try:
        payload = request.get_json(force=True)
        image_b64 = payload.get("image_base64")
        image_url = payload.get("image_url")
        mood = payload.get("mood")  # optional

        if not image_b64 and not image_url:
            return jsonify({"error": "image_base64 or image_url required"}), 400

        # Load image
        if image_b64:
            img = _decode_base64_image(image_b64)
        else:
            img = _download_image_from_url(image_url)

        # Remove background if possible
        img_nobg = _remove_background(img)

        # Resize + center to sticker canvas
        canvas = _resize_and_center(img_nobg)

        # If mood not provided, detect mood using Gemini
        detected_mood = mood
        if not mood:
            if not GEMINI_API_KEY:
                # fallback to neutral
                detected_mood = "neutral"
            else:
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    mood_prompt = (
                        "Look at this image. In one word, pick the best mood from: funny, romantic, sarcastic, motivational, cute, angry, sad, neutral. "
                        "Return only the mood word, no explanation."
                    )
                    # Send image bytes + prompt
                    buffer = io.BytesIO()
                    canvas.convert("RGB").save(buffer, format="JPEG")
                    image_bytes = buffer.getvalue()
                    response = model.generate_content([
                        {"mime_type": "image/jpeg", "data": image_bytes},
                        mood_prompt
                    ])
                    detected_mood = response.text.strip().lower().split()[0]
                except Exception as e:
                    print("Mood detection failed:", e)
                    detected_mood = "neutral"

        # Generate caption with Gemini
        if not GEMINI_API_KEY:
            # fallback: simple canned captions
            caption_map = {
                "funny": "No filter 😂",
                "romantic": "Heart eyes 💖",
                "sarcastic": "Sure, Jan",
                "motivational": "Go get it!",
                "cute": "Too adorable",
                "angry": "Not today 😤",
                "sad": "Sending hugs",
                "neutral": "Vibes"
            }
            caption = caption_map.get(detected_mood, "Vibes")
        else:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                caption_prompt = (
                    f"Generate a short (max 8 words), punchy sticker caption for this image in a {detected_mood} tone. "
                    "Keep it simple and easy to read on a sticker. Return only the caption text."
                )
                buffer = io.BytesIO()
                canvas.convert("RGB").save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()
                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": image_bytes},
                    caption_prompt
                ])
                caption = response.text.strip().split('\n')[0]
            except Exception as e:
                print("Caption generation failed:", e)
                caption = "Vibes"

        # Overlay caption onto sticker
        final_img = _overlay_caption(canvas, caption)

        # Save sticker
        filename = f"sticker_{uuid.uuid4().hex}.png"
        _save_image(final_img, filename)

        # Public URL (depends on hosting); here we return relative path
        sticker_url = request.url_root.rstrip('/') + f"/static/{filename}"

        return jsonify({
            "sticker_url": sticker_url,
            "caption": caption,
            "mood": detected_mood
        })

    except Exception as e:
        print("Error in generate_sticker:", e)
        return jsonify({"error": str(e)}), 500


# Serve static files (stickers)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/', methods=["GET"])
def index():
    return jsonify({"status": "Smart Sticker MCP server running"})


if __name__ == '__main__':
    # For local testing only. In production use a proper WSGI server.
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
