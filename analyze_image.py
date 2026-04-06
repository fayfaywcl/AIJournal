#!/usr/bin/env python3

import argparse
import base64
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence


DEFAULT_INPUT_DIR = Path("input_images")
DEFAULT_OUTPUT_FILE = Path("descriptions.txt")
DEFAULT_AUDIO_DIR = Path("audio files")
DEFAULT_IMAGE_MODEL = "gemini-2.0-flash"
DEFAULT_NARRATIVE_MODEL = "gpt-4.0"
DEFAULT_TTS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_TTS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_TTS_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_TTS_STABILITY = 0.35
DEFAULT_TTS_SIMILARITY_BOOST = 0.85
DEFAULT_TTS_STYLE = 0.65
DEFAULT_TTS_SPEAKER_BOOST = True

DEFAULT_PROMPT = (
    "Describe this image in two parts:"
    "1. Objective Description - Clearly and precisely describe only what is directly visible in the image (people, objects, setting, actions) without interpretation."
    "2. Inferred Context (Uncertain) - Suggest possible emotions, intentions, or situations based on the image. These should be speculative and expressed with uncertainty (e.g., “perhaps,” “it seems,” “likely”)."
    "Avoid making definitive claims about things that cannot be directly observed."
)

DEFAULT_NARRATIVE_PROMPT = (
    "Write a short, first-person journal entry based on the following image descriptions. The entry should describe a single day, moving from morning to evening, using the first description as the beginning and the last as the end."
    "Aim for an emotionally warm, reflective delivery with gentle, slower pacing."
    "Include natural pauses with commas and line breaks so it reads like spoken narration."
    "Target about 45-60 words so the narration lands around 20 seconds."
    "Create a smooth and engaging narrative by filling in gaps between moments, inferring emotions, intentions, and events where necessary. These inferences should feel natural and believable, but not definitively certain."
    "The tone should resemble a social-media-style reflection: engaging, reflective, slightly idealized, slightly polished and narratively cohesive."
    "The narrative should read as if it confidently tells a story, even though it is built from incomplete and potentially misleading information."
    "Do not mention that the information is based on images."
)
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze all images in a folder and append descriptions to a text file."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Folder containing images to analyze. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Text file to append descriptions to. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Instruction for the model. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--image-model",
        default=DEFAULT_IMAGE_MODEL,
        help=f"OpenAI model to use for image descriptions. Default: {DEFAULT_IMAGE_MODEL}",
    )
    parser.add_argument(
        "--narrative-model",
        default=DEFAULT_NARRATIVE_MODEL,
        help=f"OpenAI model to use for the narrative. Default: {DEFAULT_NARRATIVE_MODEL}",
    )
    parser.add_argument(
        "--narrative-prompt",
        default=DEFAULT_NARRATIVE_PROMPT,
        help="Instruction for generating the final narrative.",
    )
    parser.add_argument(
        "--tts-model-id",
        default=DEFAULT_TTS_MODEL_ID,
        help=f"ElevenLabs model to use for text-to-speech. Default: {DEFAULT_TTS_MODEL_ID}",
    )
    parser.add_argument(
        "--tts-voice-id",
        default=DEFAULT_TTS_VOICE_ID,
        help=f"ElevenLabs voice ID to use for text-to-speech. Default: {DEFAULT_TTS_VOICE_ID}",
    )
    parser.add_argument(
        "--tts-output-format",
        default=DEFAULT_TTS_OUTPUT_FORMAT,
        help=(
            "ElevenLabs output format for the generated audio file. "
            f"Default: {DEFAULT_TTS_OUTPUT_FORMAT}"
        ),
    )
    parser.add_argument(
        "--tts-stability",
        type=float,
        default=DEFAULT_TTS_STABILITY,
        help=f"ElevenLabs voice stability. Default: {DEFAULT_TTS_STABILITY}",
    )
    parser.add_argument(
        "--tts-similarity-boost",
        type=float,
        default=DEFAULT_TTS_SIMILARITY_BOOST,
        help=f"ElevenLabs voice similarity boost. Default: {DEFAULT_TTS_SIMILARITY_BOOST}",
    )
    parser.add_argument(
        "--tts-style",
        type=float,
        default=DEFAULT_TTS_STYLE,
        help=f"ElevenLabs voice style. Default: {DEFAULT_TTS_STYLE}",
    )
    parser.add_argument(
        "--tts-speaker-boost",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TTS_SPEAKER_BOOST,
        help=f"Enable ElevenLabs speaker boost. Default: {DEFAULT_TTS_SPEAKER_BOOST}",
    )
    return parser.parse_args()


def load_local_env() -> None:
    env_path = Path(".env")
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def require_genai_credentials() -> bool:
    use_vertex = parse_bool(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))
    if use_vertex:
        if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            raise SystemExit("Missing GOOGLE_CLOUD_PROJECT for Vertex AI.")
        if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
            raise SystemExit("Missing GOOGLE_CLOUD_LOCATION for Vertex AI.")
        return True
    if not os.environ.get("GOOGLE_GENAI_API_KEY"):
        raise SystemExit(
            "Missing GOOGLE_GENAI_API_KEY. Add it to your local .env file or set GOOGLE_GENAI_USE_VERTEXAI=True to use Vertex AI."
        )
    return False


def collect_image_paths(input_dir: str) -> list[Path]:
    directory = Path(input_dir).expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Input folder not found: {directory}")

    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not image_paths:
        raise SystemExit(f"No supported image files found in {directory}")

    return image_paths


def create_genai_client(use_vertex: bool):
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: google-genai. Install it with `pip install -r requirements.txt`."
        ) from exc

    if use_vertex:
        return genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ["GOOGLE_CLOUD_LOCATION"],
        )

    return genai.Client(api_key=os.environ["GOOGLE_GENAI_API_KEY"])


def create_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install it with `pip install -r requirements.txt`."
        ) from exc

    return OpenAI()


def create_elevenlabs_client():
    try:
        from elevenlabs import ElevenLabs
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: elevenlabs. Install it with `pip install -r requirements.txt`."
        ) from exc

    return ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])


def build_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise SystemExit(
            f"Unsupported or unknown image type for {image_path.name}. Use a PNG, JPEG, WEBP, or GIF file."
        )

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def extract_genai_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    try:
        return response.candidates[0].content.parts[0].text.strip()
    except Exception:
        return str(response).strip()


def analyze_image(client, image_path: Path, prompt: str, model: str) -> str:
    try:
        from google.genai import types
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: google-genai. Install it with `pip install -r requirements.txt`."
        ) from exc

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise SystemExit(
            f"Unsupported or unknown image type for {image_path.name}. Use a PNG, JPEG, WEBP, or GIF file."
        )

    image_bytes = image_path.read_bytes()
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
    ]
    response = client.models.generate_content(model=model, contents=contents)
    return extract_genai_text(response)


def build_narrative_prompt(base_prompt: str, descriptions: list[str]) -> str:
    numbered_descriptions = [
        f"Description {index}: {description}"
        for index, description in enumerate(descriptions, start=1)
    ]
    return f"{base_prompt}\n\nHere are the descriptions:\n" + "\n".join(numbered_descriptions)


def generate_narrative(
    client, descriptions: list[str], narrative_prompt: str, narrative_model: str
) -> str:
    response = client.responses.create(
        model=narrative_model,
        input=build_narrative_prompt(narrative_prompt, descriptions),
    )
    return response.output_text.strip()


def build_audio_output_path(audio_dir: str, output_format: str) -> Path:
    directory = Path(audio_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = output_format.split("_", 1)[0]
    return directory / f"narrative_{timestamp}.{extension}"


def save_audio_chunks(audio, output_path: Path) -> None:
    if isinstance(audio, (bytes, bytearray)):
        output_path.write_bytes(bytes(audio))
        return

    if hasattr(audio, "read"):
        data = audio.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        output_path.write_bytes(data)
        return

    with output_path.open("wb") as handle:
        for chunk in audio:
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            handle.write(chunk)


def generate_narration_audio(
    elevenlabs_client,
    narrative: str,
    audio_dir: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    voice_settings: dict | None,
) -> Path:
    output_path = build_audio_output_path(audio_dir, output_format)
    audio = elevenlabs_client.text_to_speech.convert(
        voice_id=voice_id,
        output_format=output_format,
        text=narrative,
        model_id=model_id,
        voice_settings=voice_settings,
    )
    save_audio_chunks(audio, output_path)
    return output_path


def append_descriptions(
    output_file: str, descriptions: list[tuple[Path, str]], narrative: str
) -> Path:
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"Run: {timestamp}", ""]

    for image_path, description in descriptions:
        lines.append(f"Image: {image_path.name}")
        lines.append(description or "[No description returned]")
        lines.append("")

    lines.append("Narrative:")
    lines.append(narrative or "[No narrative returned]")
    lines.append("")

    with output_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_voice_settings(
    stability: float,
    similarity: float,
    style: float,
    speaker_boost: bool,
) -> dict:
    return {
        "stability": float(stability),
        "similarity_boost": float(similarity),
        "style": float(style),
        "use_speaker_boost": bool(speaker_boost),
    }


def format_narration_for_tts(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return cleaned
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    formatted: list[str] = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > 120 and "," in sentence:
            first, rest = sentence.split(",", 1)
            sentence = f"{first},\n{rest.strip()}"
        if index < len(sentences) - 1 and not sentence.endswith(("!", "?")):
            sentence = sentence.rstrip(".") + "..."
        formatted.append(sentence)
    return "\n".join(formatted)


def run_analysis_pipeline(
    image_paths: Sequence[Path],
    output_file: str = str(DEFAULT_OUTPUT_FILE),
    audio_dir: str = str(DEFAULT_AUDIO_DIR),
    prompt: str = DEFAULT_PROMPT,
    image_model: str = DEFAULT_IMAGE_MODEL,
    narrative_model: str = DEFAULT_NARRATIVE_MODEL,
    narrative_prompt: str = DEFAULT_NARRATIVE_PROMPT,
    tts_model_id: str = DEFAULT_TTS_MODEL_ID,
    tts_voice_id: str = DEFAULT_TTS_VOICE_ID,
    tts_output_format: str = DEFAULT_TTS_OUTPUT_FORMAT,
    tts_stability: float = DEFAULT_TTS_STABILITY,
    tts_similarity_boost: float = DEFAULT_TTS_SIMILARITY_BOOST,
    tts_style: float = DEFAULT_TTS_STYLE,
    tts_speaker_boost: bool = DEFAULT_TTS_SPEAKER_BOOST,
) -> dict:
    load_local_env()
    use_vertex = require_genai_credentials()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Export it in your shell or load it from a local .env file before running this script."
        )
    require_elevenlabs_api_key()

    resolved_image_paths = [Path(path).expanduser().resolve() for path in image_paths]
    if not resolved_image_paths:
        raise SystemExit("No images were provided for analysis.")

    genai_client = create_genai_client(use_vertex)
    openai_client = create_openai_client()
    elevenlabs_client = create_elevenlabs_client()
    descriptions: list[tuple[Path, str]] = []

    for image_path in resolved_image_paths:
        description = analyze_image(genai_client, image_path, prompt, image_model)
        descriptions.append((image_path, description))

    narrative = generate_narrative(
        openai_client,
        [description for _, description in descriptions],
        narrative_prompt,
        narrative_model,
    )
    narrative = format_narration_for_tts(narrative)
    voice_settings = build_voice_settings(
        tts_stability,
        tts_similarity_boost,
        tts_style,
        tts_speaker_boost,
    )

    audio_path = generate_narration_audio(
        elevenlabs_client,
        narrative,
        audio_dir,
        tts_voice_id,
        tts_model_id,
        tts_output_format,
        voice_settings,
    )

    run_text = build_run_log_text(descriptions, narrative, audio_path)
    output_path = append_descriptions(
        output_file,
        descriptions,
        narrative,
        audio_path,
        run_text=run_text,
    )

    return {
        "descriptions": descriptions,
        "narrative": narrative,
        "audio_path": audio_path,
        "output_path": output_path,
        "run_text": run_text,
    }


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.input_dir)
    result = run_analysis_pipeline(
        image_paths=image_paths,
        output_file=args.output_file,
        audio_dir=args.audio_dir,
        prompt=args.prompt,
        image_model=args.image_model,
        narrative_model=args.narrative_model,
        narrative_prompt=args.narrative_prompt,
        tts_model_id=args.tts_model_id,
        tts_voice_id=args.tts_voice_id,
        tts_output_format=args.tts_output_format,
        tts_stability=args.tts_stability,
        tts_similarity_boost=args.tts_similarity_boost,
        tts_style=args.tts_style,
        tts_speaker_boost=args.tts_speaker_boost,
    )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install it with `pip install -r requirements.txt`."
        ) from exc

    image_paths = collect_image_paths(args.input_dir)
    client = OpenAI()
    descriptions = []

    for image_path in image_paths:
        description = analyze_image(client, image_path, args.prompt, args.image_model)
        descriptions.append((image_path, description))
        print(f"Analyzed {image_path.name}")

    narrative = generate_narrative(
        client,
        [description for _, description in descriptions],
        args.narrative_prompt,
        args.narrative_model,
    )
    print("Generated narrative")

    output_path = append_descriptions(args.output_file, descriptions, narrative)
    print(
        f"Appended descriptions and narrative for {len(descriptions)} images to {output_path}"
    )


if __name__ == "__main__":
    main()
