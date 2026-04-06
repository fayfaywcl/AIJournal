# CART-498-AI-Generated-Daily-Journal

<<<<<<< Updated upstream
Small Python utility for analyzing all images in `input_images`, generating a short narrative from the resulting descriptions, and appending both the descriptions and the narrative to `descriptions.txt`.
=======
Small Python utilities for:

- analyzing a single image with the OpenAI Responses API
- turning a folder of images into Runway Gen-4 Turbo video transitions
>>>>>>> Stashed changes

## Setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your API key without hardcoding it into the script:

```bash
cp .env.example .env
```

Then edit `.env` and add your real keys:

```bash
OPENAI_API_KEY=your_real_key_here
RUNWAYML_API_SECRET=your_real_runway_key_here
```

Load it into your shell:

```bash
set -a
source .env
set +a
```

`.env` is ignored by Git, so it will not be committed to GitHub unless you remove that rule.

## Run

<<<<<<< Updated upstream
Analyze every supported image in `input_images`, generate a narrative from the eight descriptions, and append the results to `descriptions.txt`:
=======
Analyze images with OpenAI:

Analyze a public image URL:
>>>>>>> Stashed changes

```bash
python3 analyze_image.py
```

Use a custom prompt:

```bash
python3 analyze_image.py \
  --prompt "Describe this image like a journal entry prompt, including mood, setting, and notable details."
```

<<<<<<< Updated upstream
Use a different folder or output file:

```bash
python3 analyze_image.py \
  --input-dir "input_images" \
  --output-file "descriptions.txt"
```

Override the image-analysis model or the narrative model:

```bash
python3 analyze_image.py \
  --image-model "gpt-4.1-mini" \
<<<<<<< Updated upstream
  --narrative-model "gpt-5.4"
=======
Generate video clips from a folder of local images:
=======
  --narrative-model "gpt-5.4" \
  --tts-voice-id "JBFqnCBsd6RMkjVDRZzb" \
  --tts-model-id "eleven_multilingual_v2" \
  --tts-stability 0.35 \
  --tts-similarity-boost 0.85 \
  --tts-style 0.65 \
  --tts-speaker-boost
```

## Generate Video

Generate Runway clips from a folder of local images:
>>>>>>> Stashed changes

```bash
python3 imagetovideo.py --input-dir "/path/to/images"
```

Generate one stitched 5-second final video for every 4 images:

```bash
python3 imagetovideo.py \
  --input-dir "/path/to/images" \
  --group-size 4 \
  --total-duration 5 \
  --stitch
>>>>>>> Stashed changes
```

## Notes

<<<<<<< Updated upstream
<<<<<<< Updated upstream
- The default image-description model is `gpt-4.1-mini`.
- The default narrative model is `gpt-5.4`.
- The script skips hidden files like `.DS_Store`.
- The script appends each run to `descriptions.txt`, so previous descriptions and narratives are preserved.
=======
- The default model is `gpt-4.1-mini`, but you can override it with `--model`.
- The script accepts exactly one image at a time.
- `imagetovideo.py` uses Runway `gen4_turbo`.
- The default prompt treats each 4-image group as reference moments from one day.
- Runway does not currently support 4 images in a single `image_to_video` generation. This script handles 4 images by generating adjacent transition clips and splitting the total requested duration across them, then optionally merging them with `ffmpeg`.
>>>>>>> Stashed changes
- Never commit your real API key, `.env`, or any file containing secrets.


Terminal command (replace the url with any image url you want)
python3 analyze_image.py --image-url "https://static.wikia.nocookie.net/obamium/images/c/cd/Screenshot_20.jpg/revision/latest?cb=20210915024847"
=======
- `analyze_image.py` uses `gpt-4.1-mini` for image descriptions and `gpt-5.4` for the final narrative by default.
- `app.py` reuses the same analysis pipeline as `analyze_image.py`, so the CLI and website produce the same saved output format.
- Generated narration audio is saved to `audio files/`.
- `descriptions.txt` keeps an append-only log of each run, including the generated narrative and saved audio path.
- Uploaded website images are stored temporarily in `web_uploads/`.
- `imagetovideo.py` uses Runway `gen4_turbo` by default.
- If `imagetovideo.py` is run without `--prompt`, it uses the latest `Narrative:` block from `descriptions.txt`.
- `director_pipeline.py` uses OpenAI to create image analyses and a shot plan, then calls Veo through Vertex AI using `google-genai`.
- `director_pipeline.py` targets a 20 second total duration by default, matching the narration length guidance.
- ElevenLabs voice settings can be supplied via CLI (`analyze_image.py`) or environment variables (`director_pipeline.py`).
- Veo outputs are written to the Cloud Storage prefix configured by `GOOGLE_CLOUD_VEO_OUTPUT_GCS_URI`, then downloaded back into the local run folder.
- `elevenlabs` is optional for `director_pipeline.py` when you run with `--skip-audio`.
- Never commit your real API keys, `.env`, or any file containing secrets.
>>>>>>> Stashed changes
