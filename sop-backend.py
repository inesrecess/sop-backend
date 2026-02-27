# backend/main.py
# Deploy this on Railway, Render, or any Python host
# Requirements: fastapi, uvicorn, yt-dlp, ffmpeg-python, anthropic, python-multipart

import os, base64, tempfile, subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import anthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

class ExtractRequest(BaseModel):
    video_url: str
    timestamps: list[str]  # ["0:15", "1:02", "2:45", ...]

class AnalyzeRequest(BaseModel):
    video_url: str
    notes: str = ""

# ── 1. Auto-detect key moments from video metadata + notes ──────────────────
@app.post("/analyze-video")
async def analyze_video(req: AnalyzeRequest):
    """
    Step 1: Claude suggests key timestamps based on notes + video duration.
    Returns list of {timestamp, reason} for frame extraction.
    """
    # Get video duration via yt-dlp (works for Loom, YouTube, Vimeo)
    try:
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(req.video_url, download=False)
            duration = info.get("duration", 120)  # seconds
            title = info.get("title", "")
    except Exception as e:
        raise HTTPException(400, f"Could not read video: {e}")

    # Ask Claude to suggest key moment timestamps
    prompt = f"""A process video is {duration} seconds long. Title: "{title}".
Process notes: {req.notes or "Not provided"}

Suggest 6-10 key timestamps (evenly distributed but weighted toward action moments) 
to extract screenshot frames that would best illustrate an SOP.

Return ONLY valid JSON (no backticks):
{{
  "timestamps": [
    {{"time": "0:05", "seconds": 5, "reason": "Opening state / starting point"}},
    {{"time": "0:30", "seconds": 30, "reason": "First key action"}},
    ...
  ]
}}"""

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    import json
    text = resp.content[0].text.replace("```json","").replace("```","").strip()
    return json.loads(text)


# ── 2. Extract frames at given timestamps using yt-dlp + FFmpeg ─────────────
@app.post("/extract-frames")
async def extract_frames(req: ExtractRequest):
    """
    Step 2: Download video, extract frames at each timestamp.
    Returns list of base64-encoded JPEG images.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")

        # Download video (yt-dlp handles Loom, YouTube, Vimeo, etc.)
        ydl_opts = {
            "outtmpl": video_path,
            "format": "mp4/bestvideo[ext=mp4]",
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([req.video_url])
        except Exception as e:
            raise HTTPException(400, f"Video download failed: {e}")

        frames = []
        for ts in req.timestamps:
            # Convert MM:SS to seconds
            parts = ts.split(":")
            secs = int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else int(parts[0])
            out_path = os.path.join(tmpdir, f"frame_{secs}.jpg")

            # Extract frame with FFmpeg
            result = subprocess.run([
                "ffmpeg", "-ss", str(secs), "-i", video_path,
                "-frames:v", "1", "-q:v", "2",
                "-vf", "scale=1280:-1",  # HD width, auto height
                out_path, "-y", "-loglevel", "quiet"
            ], capture_output=True)

            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                frames.append({"timestamp": ts, "seconds": secs, "image_b64": b64})
            else:
                frames.append({"timestamp": ts, "seconds": secs, "image_b64": None, "error": "Frame extraction failed"})

        return {"frames": frames}


# ── 3. Claude analyzes each frame and describes what it shows ────────────────
@app.post("/describe-frames")
async def describe_frames(body: dict):
    """
    Step 3: Send each frame to Claude Vision to get a description
    of what's happening on screen at that moment.
    """
    frames = body.get("frames", [])
    described = []

    for frame in frames:
        if not frame.get("image_b64"):
            described.append({**frame, "description": "Frame not available"})
            continue

        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame["image_b64"]
                        }
                    },
                    {
                        "type": "text",
                        "text": "This is a frame from a business process tutorial video. In 1-2 sentences, describe exactly what is visible on screen and what action is being performed or demonstrated. Be specific about UI elements, buttons, fields, or content visible."
                    }
                ]
            }]
        )
        described.append({
            **frame,
            "description": resp.content[0].text
        })

    return {"frames": described}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
