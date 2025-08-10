import os
import io
import uuid
import base64
import logging
from typing import Optional

import requests
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, constr
from pydantic import BaseModel, validator
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

# -----------------
# Configure logging
# -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------
# Load env vars
# --------------
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

REMOVEBG_API_KEY = os.getenv("REMOVEBG_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_PHONE = os.getenv("OWNER_PHONE", "919440473900")
VALIDATE_TOKEN = os.getenv("VALIDATE_TOKEN", "stickypuch")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set. AI features disabled.")

if not REMOVEBG_API_KEY:
    logger.warning("REMOVEBG_API_KEY not set. Background removal disabled.")

# -------------------
# FastAPI initialization
# -------------------
app = FastAPI(title="Smart Sticker Maker MCP Server")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --------------
# Pydantic models
# --------------

class ValidateRequest(BaseModel):
    bearer_token: constr(min_length=5)

class StickerRequest(BaseModel):
    mood: Optional[str] = None

    @validator('mood')
    def mood_must_be_valid(cls, v):
        allowed = {"funny", "romantic", "sarcastic", "motivational", "cute", "angry", "sad", "neutral"}
        if v is not None and v not in allowed:
            raise ValueError(f"mood must be one of {allowed}")
        return v


# --------------
# Helper functions
# --------------

def save_image(img: Image.Image, filename: str) -> str:
    path = os.path.join(STATIC_DIR, filename)
    img.save(path, format="PNG")
    logger.info(f"Saved sticker image: {path}")
    return path

def download_image_from_url(url: str) -> Image.Image:
    logger.info(f"Downloading image from URL: {url}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGBA")

def decode_base64_image(b64str: str) -> Image.Image:
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert("RGBA")

def remove_background(img: Image.Image) -> Image.Image:
    if not REMOVEBG_API_KEY:
        logger.info("No remove.bg API key provided; skipping background removal.")
        return img

    try:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)

        response = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": buffered},
            data={"size": "auto"},
            headers={"X-Api-Key": REMOVEBG_API_KEY},
            timeout=60
        )
        if response.status_code == 200:
            logger.info("Background removed successfully.")
            return Image.open(io.BytesIO(response.content)).convert("RGBA")
        else:
            logger.warning(f"remove.bg API error {response.status_code}: {response.text}")
            return img
    except Exception as e:
        logger.error(f"Background removal failed: {e}")
        return img

def resize_and_center(img: Image.Image, size=(512, 512)) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    img.thumbnail((size[0] - 20, size[1] - 100), Image.LANCZOS)
    x = (size[0] - img.width) // 2
    y = max(10, (size[1] - img.height) // 2)
    canvas.paste(img, (x, y), img)
    return canvas

def overlay_caption(img: Image.Image, caption: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    width, height = img.size
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 32)
    except Exception:
        font = ImageFont.load_default()

    max_width = width - 40
    words = caption.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if draw.textsize(test_line, font=font)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    total_text_height = sum(draw.textsize(line, font=font)[1] + 6 for line in lines)
    y_text = height - total_text_height - 20

    # Semi-transparent background for text
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(20, y_text - 10), (width - 20, height - 10)], fill=(0, 0, 0, 120))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)
    for line in lines:
        w, h = draw.textsize(line, font=font)
        x_text = (width - w) // 2
        # Outline
        for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            draw.text((x_text+dx, y_text+dy), line, font=font, fill=(0,0,0,255))
        # Main text
        draw.text((x_text, y_text), line, font=font, fill=(255,255,255,255))
        y_text += h + 6

    return img

def detect_mood_with_ai(img: Image.Image) -> str:
    if not GEMINI_API_KEY:
        return "neutral"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Look at this image. In one word, pick the best mood from: funny, romantic, sarcastic, motivational, cute, angry, sad, neutral."
        )
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": buf.getvalue()},
            prompt
        ])
        mood = response.text.strip().lower().split()[0]
        if mood not in {"funny","romantic","sarcastic","motivational","cute","angry","sad","neutral"}:
            mood = "neutral"
        return mood
    except Exception as e:
        logger.error(f"Mood detection failed: {e}")
        return "neutral"

def generate_caption_with_ai(img: Image.Image, mood: str) -> str:
    if not GEMINI_API_KEY:
        return None

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Generate a short (max 8 words), punchy sticker caption for this image in a {mood} tone. "
            "Return only the caption text."
        )
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": buf.getvalue()},
            prompt
        ])
        caption = response.text.strip().split('\n')[0]
        return caption if caption else None
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        return None

# Fallback captions for moods
CAPTION_FALLBACKS = {
    "funny": "No filter 😂",
    "romantic": "Heart eyes 💖",
    "sarcastic": "Sure, Jan",
    "motivational": "Go get it!",
    "cute": "Too adorable",
    "angry": "Not today 😤",
    "sad": "Sending hugs",
    "neutral": "Vibes"
}

# --------------
# API Endpoints
# --------------

from fastapi.responses import PlainTextResponse

@app.post("/validate")
async def validate(data: ValidateRequest):
    if data.bearer_token == VALIDATE_TOKEN:
        return PlainTextResponse(str(OWNER_PHONE))
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")

@app.get("/mcp")
@app.post("/mcp")
async def mcp_handler(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    return JSONResponse({
        "status": "success",
        "message": "MCP endpoint connected successfully",
        "received": data
    })

@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    return {
        "name": "Smart Sticker MCP",
        "version": "1.0",
        "tools": [
            {
                "name": "validate",
                "description": "Validate token and return owner phone",
                "input_schema": {
                    "type": "object",
                    "properties": {"bearer_token": {"type": "string"}},
                    "required": ["bearer_token"]
                }
            },
            {
                "name": "generate_sticker",
                "description": "Generate a mood-based sticker with optional caption",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_base64": {"type": "string"},
                        "image_url": {"type": "string"},
                        "mood": {
                            "type": "string",
                            "enum": [
                                "funny", "romantic", "sarcastic",
                                "motivational", "cute", "angry",
                                "sad", "neutral"
                            ]
                        }
                    },
                    "anyOf": [
                        {"required": ["image_base64"]},
                        {"required": ["image_url"]}
                    ]
                }
            }
        ]
    }

@app.post("/generate_sticker")
async def generate_sticker(payload: StickerRequest, request: Request):
    if not payload.image_base64 and not payload.image_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="image_base64 or image_url required")

    # Load image
    try:
        img = decode_base64_image(payload.image_base64) if payload.image_base64 else download_image_from_url(payload.image_url)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data or URL")

    # Remove background
    img_nobg = remove_background(img)

    # Resize and center
    canvas = resize_and_center(img_nobg)

    # Mood detection
    mood = payload.mood
    if not mood:
        mood = detect_mood_with_ai(canvas)

    # Caption generation
    caption = None
    if GEMINI_API_KEY:
        caption = generate_caption_with_ai(canvas, mood)

    if not caption:
        caption = CAPTION_FALLBACKS.get(mood, "Vibes")

    # Overlay caption on image
    final_img = overlay_caption(canvas, caption)

    # Save final image
    filename = f"sticker_{uuid.uuid4().hex}.png"
    save_image(final_img, filename)

    sticker_url = str(request.base_url).rstrip("/") + f"/static/{filename}"

    return {
        "sticker_url": sticker_url,
        "caption": caption,
        "mood": mood
    }

@app.get("/")
async def root():
    return {"status": "Smart Sticker MCP server running"}

# --------------
# Run the app
# --------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
