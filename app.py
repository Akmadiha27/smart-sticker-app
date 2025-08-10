import os
import io
import uuid
import base64
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

MY_NUMBER = "919440473900"

# Optional: rembg for background removal
# remove.bg API key from env
REMOVEBG_API_KEY = os.getenv("REMOVEBG_API_KEY")
if not REMOVEBG_API_KEY:
    print("WARNING: REMOVEBG_API_KEY not set.")


# FastAPI instance
app = FastAPI(title="Smart Sticker Maker MCP Server")

# Static directory setup
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_PHONE = os.getenv("OWNER_PHONE", "919440473900")
VALIDATE_TOKEN = os.getenv("VALIDATE_TOKEN", "stickypuch")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not set.")

# ------------------
# Helper functions
# ------------------

def _save_image(img: Image.Image, filename: str) -> str:
    path = os.path.join(STATIC_DIR, filename)
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
    """
    Uses remove.bg API to remove background from an image.
    """
    if not REMOVEBG_API_KEY:
        print("No remove.bg API key provided. Returning original image.")
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
            return Image.open(io.BytesIO(response.content)).convert("RGBA")
        else:
            print(f"remove.bg API failed: {response.status_code} - {response.text}")
            return img
    except Exception as e:
        print("remove.bg request failed:", e)
        return img


def _resize_and_center(img: Image.Image, size=(512, 512)) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    img.thumbnail((size[0] - 20, size[1] - 100), Image.LANCZOS)
    x = (size[0] - img.width) // 2
    y = max(10, (size[1] - img.height) // 2)
    canvas.paste(img, (x, y), img)
    return canvas

def _overlay_caption(img: Image.Image, caption: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    width, height = img.size
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font = ImageFont.truetype(font_path, 32)
    except Exception:
        font = ImageFont.load_default()

    max_width = width - 40
    lines, current = [], ""
    for w in caption.split():
        test = (current + " " + w).strip()
        if draw.textsize(test, font=font)[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    total_text_height = sum(draw.textsize(line, font=font)[1] + 6 for line in lines)
    y_text = height - total_text_height - 20

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(20, y_text - 10), (width - 20, height - 10)], fill=(0, 0, 0, 120))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    for line in lines:
        w, h = draw.textsize(line, font=font)
        x_text = (width - w) // 2
        for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            draw.text((x_text+dx, y_text+dy), line, font=font, fill=(0,0,0,255))
        draw.text((x_text, y_text), line, font=font, fill=(255,255,255,255))
        y_text += h + 6

    return img

# ------------------
# Pydantic models
# ------------------

class ValidateRequest(BaseModel):
    bearer_token: str

class StickerRequest(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    mood: str | None = None

# ------------------
# Endpoints
# ------------------

@app.post("/validate")
async def validate(data: ValidateRequest):
    if data.bearer_token == VALIDATE_TOKEN:
        return MY_NUMBER
    return JSONResponse(status_code=403, content={"error": "invalid token"})

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
                    "properties": {
                        "bearer_token": {"type": "string"}
                    },
                    "required": ["bearer_token"]
                }
            }
        ]
    }

@app.post("/generate_sticker")
async def generate_sticker(payload: StickerRequest, request: Request):
    try:
        if not payload.image_base64 and not payload.image_url:
            return JSONResponse(status_code=400, content={"error": "image_base64 or image_url required"})

        if payload.image_base64:
            img = _decode_base64_image(payload.image_base64)
        else:
            img = _download_image_from_url(payload.image_url)

        img_nobg = _remove_background(img)
        canvas = _resize_and_center(img_nobg)

        detected_mood = payload.mood
        if not detected_mood:
            if GEMINI_API_KEY:
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    mood_prompt = (
                        "Look at this image. In one word, pick the best mood from: funny, romantic, sarcastic, motivational, cute, angry, sad, neutral."
                    )
                    buf = io.BytesIO()
                    canvas.convert("RGB").save(buf, format="JPEG")
                    response = model.generate_content([
                        {"mime_type": "image/jpeg", "data": buf.getvalue()},
                        mood_prompt
                    ])
                    detected_mood = response.text.strip().lower().split()[0]
                except Exception as e:
                    print("Mood detection failed:", e)
                    detected_mood = "neutral"
            else:
                detected_mood = "neutral"

        if GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                caption_prompt = (
                    f"Generate a short (max 8 words), punchy sticker caption for this image in a {detected_mood} tone. "
                    "Return only the caption text."
                )
                buf = io.BytesIO()
                canvas.convert("RGB").save(buf, format="JPEG")
                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": buf.getvalue()},
                    caption_prompt
                ])
                caption = response.text.strip().split('\n')[0]
            except Exception as e:
                print("Caption generation failed:", e)
                caption = "Vibes"
        else:
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

        final_img = _overlay_caption(canvas, caption)
        filename = f"sticker_{uuid.uuid4().hex}.png"
        _save_image(final_img, filename)

        sticker_url = str(request.base_url).rstrip("/") + f"/static/{filename}"

        return {
            "sticker_url": sticker_url,
            "caption": caption,
            "mood": detected_mood
        }

    except Exception as e:
        print("Error in generate_sticker:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def index():
    return {"status": "Smart Sticker MCP server running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
