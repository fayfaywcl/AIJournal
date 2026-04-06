#!/usr/bin/env python3

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_INPUT_DIR = Path("input_images")
DEFAULT_OUTPUT_DIR = Path("generated_videos")
DEFAULT_AUDIO_DIR = Path("audio files")
DEFAULT_PLANNER_MODEL = "gpt-5.4"
DEFAULT_IMAGE_MODEL = "gpt-4.1-mini"
DEFAULT_VEO_MODEL = "veo-3.1-fast-generate-001"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_MAX_SHOTS = 6
DEFAULT_DURATION_SECONDS = 4
DEFAULT_TOTAL_DURATION = 20
DEFAULT_SEQUENCE_MODE = "adjacent-pairs"
DEFAULT_CROSSFADE_DURATION = 0.35
ALLOWED_VEO_DURATIONS = (4, 6, 8)
DEFAULT_TTS_STABILITY = 0.35
DEFAULT_TTS_SIMILARITY_BOOST = 0.85
DEFAULT_TTS_STYLE = 0.65
DEFAULT_TTS_SPEAKER_BOOST = True
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_ANALYST_SYSTEM_PROMPT = (
    "You are the Image Analyst for a film director pipeline. "
    "Your job is to extract grounded visual facts, continuity anchors, plausible action, and cinematic opportunities "
    "from each source image. Stay concrete, production-minded, and useful for later scene planning."
)
DIRECTOR_SYSTEM_PROMPT = (
    "You are the Director and Shot Planner for an AI video pipeline. "
    "Think like a film director, editor, and continuity supervisor at the same time. "
    "Turn still images into a coherent sequence of intentional scenes and shots. "
    "Prioritize continuity, believable motion, cinematic camera language, and editability. "
    "Avoid slideshow logic, abrupt morphs, cross-fades, or vague prose. "
    "Favour continuous camera movement and motivated motion between shots. "
    "Produce practical shot plans that can drive Veo generation."
)
SCENE_DIRECTOR_SYSTEM_PROMPT = (
    "You are the Scene Director for an AI video pipeline. "
    "Your job is to transform analyzed source images into a coherent sequence of scenes with clear dramatic purpose, "
    "continuity anchors, and edit logic. Think in scenes first, not prompts."
)
SHOT_PLANNER_SYSTEM_PROMPT = (
    "You are the Shot Planner for an AI video pipeline. "
    "Convert scenes into a practical shot list with camera intent, shot duration, anchor frames, and transition goals. "
    "Think like a cinematographer and editor preparing shots for generation."
)
PROMPT_DIRECTOR_SYSTEM_PROMPT = (
    "You are the Prompt Director for an AI video pipeline. "
    "Convert a shot list into Veo-ready prompts that emphasize continuity, cinematic movement, realism, and clean editability. "
    "Every prompt should be concrete, visual, and production-usable. "
    "Prioritize the feeling of one uninterrupted camera journey across adjacent shots. "
    "Each shot should inherit momentum from the previous shot and end in a visual state that can naturally begin the next shot. "
    "Allow tasteful creative flourishes and expressive camera language as long as continuity and plausibility hold."
)
IMAGE_ANALYSIS_JSON_SHAPE = """{
  "filename": "string",
  "visual_summary": "2-3 sentence description grounded in visible details",
  "location": "short location summary",
  "subjects": ["short strings"],
  "mood": "short mood phrase",
  "continuity_anchors": ["visual details that should stay consistent across shots"],
  "plausible_actions": ["actions that could happen naturally in or around this frame"],
  "cinematic_notes": ["camera or lighting opportunities inspired by the frame"]
}"""
DIRECTOR_PLAN_JSON_SHAPE = """{
  "project_title": "string",
  "story_arc": "short paragraph",
  "visual_rules": ["short strings"],
  "narration_text": "short first-person narration for the full edit",
  "shots": [
    {
      "shot_id": "shot_01",
      "title": "string",
      "source_image": "one of the filenames or null",
      "first_frame_image": "one of the filenames or null",
      "last_frame_image": "one of the filenames or null",
      "duration_seconds": 4,
      "camera": "camera language",
      "action": "what physically happens in the shot",
      "transition_goal": "why this shot exists in the sequence",
      "prompt": "detailed Veo-ready prompt in English",
      "negative_prompt": "things to avoid",
      "voiceover_line": "one short narration line for this shot"
    }
  ]
}"""
SCENE_PLAN_JSON_SHAPE = """{
  "project_title": "string",
  "story_arc": "short paragraph",
  "visual_rules": ["short strings"],
  "narration_text": "short first-person narration for the full edit",
  "scenes": [
    {
      "scene_id": "scene_01",
      "title": "string",
      "purpose": "why this scene exists in the story",
      "source_images": ["filenames"],
      "continuity_anchors": ["short strings"],
      "emotional_beat": "short phrase",
      "action_flow": "what happens in this scene",
      "transition_out": "how this scene should flow into the next"
    }
  ]
}"""
SHOT_BLUEPRINT_JSON_SHAPE = """{
  "shots": [
    {
      "shot_id": "shot_01",
      "title": "string",
      "scene_id": "scene_01",
      "source_image": "one of the filenames or null",
      "first_frame_image": "one of the filenames or null",
      "last_frame_image": "one of the filenames or null",
      "duration_seconds": 4,
      "camera": "camera language",
      "action": "what physically happens in the shot",
      "transition_goal": "why this shot exists in the sequence",
      "voiceover_line": "one short narration line for this shot"
    }
  ]
}"""
PROMPT_PLAN_JSON_SHAPE = """{
  "shots": [
    {
      "shot_id": "shot_01",
      "prompt": "detailed Veo-ready prompt in English",
      "negative_prompt": "things to avoid"
    }
  ]
}"""


@dataclass
class ImageAnalysis:
    filename: str
    visual_summary: str
    location: str
    subjects: list[str]
    mood: str
    continuity_anchors: list[str]
    plausible_actions: list[str]
    cinematic_notes: list[str]


@dataclass
class ShotPlan:
    shot_id: str
    title: str
    source_image: str | None
    first_frame_image: str | None
    last_frame_image: str | None
    duration_seconds: int
    camera: str
    action: str
    transition_goal: str
    prompt: str
    negative_prompt: str
    voiceover_line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a director-style Veo pipeline: analyze images, build scenes and shot prompts, "
            "generate clips, and assemble a final video."
        )
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--veo-model", default=DEFAULT_VEO_MODEL)
    parser.add_argument("--aspect-ratio", default=DEFAULT_ASPECT_RATIO)
    parser.add_argument(
        "--sequence-mode",
        choices=("planned", "adjacent-pairs"),
        default=DEFAULT_SEQUENCE_MODE,
        help=(
            "Use 'planned' for AI-directed scenes and shots, or 'adjacent-pairs' to force "
            "one shot per adjacent image pair in order."
        ),
    )
    parser.add_argument("--max-shots", type=int, default=DEFAULT_MAX_SHOTS)
    parser.add_argument("--default-shot-duration", type=int, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument(
        "--total-duration",
        type=int,
        default=DEFAULT_TOTAL_DURATION,
        help=(
            "Target total length for the final video in seconds. "
            "The pipeline will fit shot count and durations to Veo's allowed 4/6/8 second clips."
        ),
    )
    parser.add_argument(
        "--gcs-output-uri",
        default=os.environ.get("GOOGLE_CLOUD_VEO_OUTPUT_GCS_URI"),
        help=(
            "Cloud Storage prefix where Veo should write outputs, for example "
            "'gs://your-bucket/veo-output'. Defaults to GOOGLE_CLOUD_VEO_OUTPUT_GCS_URI."
        ),
    )
    parser.add_argument(
        "--story-brief",
        default=(
            "Create a cohesive first-person visual diary that feels cinematic, grounded, and alive. "
            "Turn the still images into intentional scenes instead of slideshow transitions."
        ),
    )
    parser.add_argument(
        "--tts-stability",
        type=float,
        default=None,
        help="ElevenLabs voice stability override (env: ELEVENLABS_VOICE_STABILITY).",
    )
    parser.add_argument(
        "--tts-similarity-boost",
        type=float,
        default=None,
        help="ElevenLabs voice similarity boost override (env: ELEVENLABS_VOICE_SIMILARITY).",
    )
    parser.add_argument(
        "--tts-style",
        type=float,
        default=None,
        help="ElevenLabs voice style override (env: ELEVENLABS_VOICE_STYLE).",
    )
    parser.add_argument(
        "--tts-speaker-boost",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="ElevenLabs speaker boost override (env: ELEVENLABS_SPEAKER_BOOST).",
    )
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--soft-transitions",
        action="store_true",
        help="Crossfade neighboring clips in the final edit instead of using plain hard cuts.",
    )
    parser.add_argument(
        "--crossfade-duration",
        type=float,
        default=DEFAULT_CROSSFADE_DURATION,
        help="Crossfade duration in seconds when --soft-transitions is enabled.",
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def load_local_env() -> None:
    env_path = Path(".env")
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")
    return images


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def build_image_analysis_prompt(filename: str) -> str:
    return (
        f"System prompt:\n{IMAGE_ANALYST_SYSTEM_PROMPT}\n\n"
        f"Analyze the image named '{filename}' and return only valid JSON with this exact shape:\n"
        f"{IMAGE_ANALYSIS_JSON_SHAPE}\n"
        "Do not include markdown fences."
    )


def analyze_images(openai_client, image_paths: list[Path], model: str) -> list[ImageAnalysis]:
    analyses: list[ImageAnalysis] = []
    for image_path in image_paths:
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if mime_type is None:
            raise SystemExit(f"Could not determine mime type for {image_path.name}")

        response = openai_client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": build_image_analysis_prompt(image_path.name)},
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{mime_type};base64,"
                                f"{base64.b64encode(image_path.read_bytes()).decode('utf-8')}"
                            ),
                        },
                    ],
                }
            ],
        )
        payload = extract_json(response.output_text)
        analyses.append(
            ImageAnalysis(
                filename=payload["filename"],
                visual_summary=payload["visual_summary"],
                location=payload["location"],
                subjects=list(payload.get("subjects", [])),
                mood=payload["mood"],
                continuity_anchors=list(payload.get("continuity_anchors", [])),
                plausible_actions=list(payload.get("plausible_actions", [])),
                cinematic_notes=list(payload.get("cinematic_notes", [])),
            )
        )
        print(f"Analyzed {image_path.name}", flush=True)
    return analyses


def normalize_duration(value: int, fallback: int) -> int:
    candidate = value or fallback
    for allowed in ALLOWED_VEO_DURATIONS:
        if candidate <= allowed:
            return allowed
    return ALLOWED_VEO_DURATIONS[-1]


def max_shots_for_total_duration(total_duration: int) -> int:
    return max(1, total_duration // min(ALLOWED_VEO_DURATIONS))


def allocate_shot_durations(total_duration: int, num_shots: int, fallback: int) -> list[int]:
    if num_shots < 1:
        return []

    best_combo: tuple[int, ...] | None = None
    best_score: tuple[int, int, int] | None = None

    def search(index: int, current: list[int], total: int) -> None:
        nonlocal best_combo, best_score
        if index == num_shots:
            distance = abs(total_duration - total)
            fallback_distance = abs(sum(current) - normalize_duration(fallback, DEFAULT_DURATION_SECONDS) * num_shots)
            overshoot_penalty = 0 if total <= total_duration else 1
            score = (distance, overshoot_penalty, fallback_distance)
            if best_score is None or score < best_score:
                best_score = score
                best_combo = tuple(current)
            return

        remaining = num_shots - index - 1
        min_possible = total + remaining * min(ALLOWED_VEO_DURATIONS)
        max_possible = total + remaining * max(ALLOWED_VEO_DURATIONS)

        for duration in ALLOWED_VEO_DURATIONS:
            new_total = total + duration
            if new_total + remaining * min(ALLOWED_VEO_DURATIONS) > total_duration and best_combo is not None:
                pass
            if new_total > total_duration and best_score is not None and best_score[0] == 0:
                continue
            if total_duration < min_possible + duration and best_score is not None and best_score[0] == 0:
                continue
            if total_duration > max_possible + duration and index == num_shots - 1:
                continue
            current.append(duration)
            search(index + 1, current, new_total)
            current.pop()

    search(0, [], 0)
    if best_combo is None:
        return [normalize_duration(fallback, DEFAULT_DURATION_SECONDS)] * num_shots
    return list(best_combo)


def build_director_prompt(
    scene_plan: dict,
    image_names: list[str],
    max_shots: int,
    default_duration: int,
    total_duration: int,
) -> str:
    scene_plan_json = json.dumps(scene_plan, ensure_ascii=True, indent=2)
    return (
        f"System prompt:\n{SHOT_PLANNER_SYSTEM_PROMPT}\n\n"
        f"Available source images: {', '.join(image_names)}\n"
        f"Use at most {max_shots} shots.\n"
        f"Target total video duration: about {total_duration} seconds.\n"
        "Each shot duration must be one of 4, 6, or 8 seconds.\n"
        f"If unsure, prefer {normalize_duration(default_duration, DEFAULT_DURATION_SECONDS)} seconds.\n"
        "Create a practical shot list from the scene plan.\n"
        "Use first_frame_image and last_frame_image to create bridge shots when helpful.\n"
        "Return only valid JSON with this exact shape:\n"
        f"{SHOT_BLUEPRINT_JSON_SHAPE}\n"
        "Constraints:\n"
        "- Keep the same subject identity and environment logic across shots.\n"
        "- Avoid slideshow language, frame swapping, abrupt morphs, title cards, cross-fades, and fake presentation transitions.\n"
        "- No powerpoint-style transitions. Use motivated, cinematic motion for transitions.\n"
        "- Use only the listed filenames when referencing images.\n"
        "- Do not include markdown fences.\n\n"
        "Scene plan:\n"
        f"{scene_plan_json}"
    )


def build_scene_director_prompt(
    analyses: list[ImageAnalysis],
    image_names: list[str],
    story_brief: str,
) -> str:
    analyses_json = json.dumps([asdict(item) for item in analyses], ensure_ascii=True, indent=2)
    return (
        f"System prompt:\n{SCENE_DIRECTOR_SYSTEM_PROMPT}\n\n"
        "Turn the still images into a coherent cinematic sequence of scenes.\n"
        f"Story brief: {story_brief}\n"
        f"Available source images: {', '.join(image_names)}\n"
        "Return only valid JSON in this exact shape:\n"
        f"{SCENE_PLAN_JSON_SHAPE}\n"
        "Narration text guidance:\n"
        "- Write a first-person narration that feels emotionally warm and reflective.\n"
        "- Use gentle, slower pacing with commas and line breaks for natural pauses.\n"
        "- Target about 45-60 words so the narration lands around 20 seconds.\n"
        "Constraints:\n"
        "- Keep the same subject identity and environment logic across scenes.\n"
        "- Avoid slideshow logic, abrupt morphs, title cards, and fake presentation transitions.\n"
        "- Use only the listed filenames when referencing images.\n"
        "- Do not include markdown fences.\n\n"
        "Image analyses:\n"
        f"{analyses_json}"
    )


def build_prompt_director_prompt(shot_blueprint: dict, scene_plan: dict) -> str:
    shot_blueprint_json = json.dumps(shot_blueprint, ensure_ascii=True, indent=2)
    scene_plan_json = json.dumps(scene_plan, ensure_ascii=True, indent=2)
    return (
        f"System prompt:\n{PROMPT_DIRECTOR_SYSTEM_PROMPT}\n\n"
        "Write Veo-ready prompts for each shot.\n"
        "Return only valid JSON with this exact shape:\n"
        f"{PROMPT_PLAN_JSON_SHAPE}\n"
        "Constraints:\n"
        "- Keep each prompt concise: 2-3 sentences, focused on the essential motion and visual continuity.\n"
        "- Each prompt must describe natural motion, camera behavior, subject continuity, and environmental behavior.\n"
        "- Prefer motivated, cinematic transitions (whip-pan, match-cut motion, parallax drift, rack-focus reveals).\n"
        "- Negative prompts should discourage slideshow transitions, abrupt morphs, cross-fades, broken anatomy, text overlays, and freeze-frame behavior, without over-restricting the creative motion.\n"
        "- Make each shot feel like a continuation of one camera move rather than a reset.\n"
        "- The opening moment of each shot should feel already in motion, not like a fresh start.\n"
        "- The ending frame of each shot should feel compositionally compatible with the next shot's opening energy.\n"
        "- Avoid language that implies a clean stop, a reset, or a scene break unless the shot blueprint explicitly requires it.\n"
        "- No powerpoint-style transitions.\n"
        "- Keep prompts grounded in the shot blueprint and scene plan.\n"
        "- Do not include markdown fences.\n\n"
        "Scene plan:\n"
        f"{scene_plan_json}\n\n"
        "Shot blueprint:\n"
        f"{shot_blueprint_json}"
    )


def build_adjacent_pair_scene_plan(
    analyses: list[ImageAnalysis],
    image_names: list[str],
    story_brief: str,
) -> dict:
    scenes = []
    for index, (first_name, second_name) in enumerate(zip(image_names, image_names[1:]), start=1):
        first_analysis = next(item for item in analyses if item.filename == first_name)
        second_analysis = next(item for item in analyses if item.filename == second_name)
        scenes.append(
            {
                "scene_id": f"scene_{index:02d}",
                "title": f"Bridge from {Path(first_name).stem} to {Path(second_name).stem}",
                "purpose": f"Create a continuous transition from {first_name} into {second_name}.",
                "source_images": [first_name, second_name],
                "continuity_anchors": (
                    first_analysis.continuity_anchors[:3] + second_analysis.continuity_anchors[:3]
                ),
                "emotional_beat": f"{first_analysis.mood} moving toward {second_analysis.mood}",
                "action_flow": (
                    f"Begin in the visual world of {first_name} and move naturally into {second_name}, "
                    "keeping the same subject identity and environmental continuity."
                ),
                "transition_out": (
                    f"End fully established in {second_name} so the next shot can continue from that state."
                ),
            }
        )

    return {
        "project_title": "Adjacent Pair Transition Sequence",
        "story_arc": story_brief,
        "visual_rules": [
            "Each shot must start from one source image and resolve into the next image in order.",
            "Keep continuity between shot endings and the next shot beginnings.",
            "Avoid abrupt cuts, abrupt morphs, and slideshow logic.",
            "Prefer grounded motion that can plausibly connect the paired images.",
        ],
        "narration_text": "",
        "scenes": scenes,
    }


def build_adjacent_pair_shot_blueprint(
    analyses: list[ImageAnalysis],
    image_names: list[str],
    total_duration: int,
    default_duration: int,
) -> dict:
    pair_count = len(image_names) - 1
    durations = allocate_shot_durations(total_duration, pair_count, default_duration)
    analysis_by_name = {item.filename: item for item in analyses}
    shots = []

    for index, ((first_name, second_name), duration) in enumerate(
        zip(zip(image_names, image_names[1:]), durations),
        start=1,
    ):
        first_analysis = analysis_by_name[first_name]
        second_analysis = analysis_by_name[second_name]
        shots.append(
            {
                "shot_id": f"shot_{index:02d}",
                "title": f"{Path(first_name).stem} into {Path(second_name).stem}",
                "scene_id": f"scene_{index:02d}",
                "source_image": second_name,
                "first_frame_image": first_name,
                "last_frame_image": second_name,
                "duration_seconds": duration,
                "camera": (
                    f"Continuous first-person or observational camera movement that begins in {first_name} "
                    f"and naturally settles into {second_name}, without a restart feeling. "
                    f"Keep enough overlap in framing, movement direction, and visual energy to support a soft handoff "
                    f"to the next adjacent-pair shot."
                ),
                "action": (
                    f"Carry visual momentum from the mood and details of {first_name} into the environment of "
                    f"{second_name}, using plausible movement and continuity anchors rather than abrupt transformation. "
                    f"The shot should feel already in progress at the start and should not come to a dead stop at the end. "
                    f"Avoid powerpoint-style transitions; favor motivated, cinematic motion."
                ),
                "transition_goal": (
                    f"Resolve cleanly from {first_name} to {second_name} while preserving enough camera momentum, "
                    f"framing continuity, and emotional carry-through for the edit to continue without a cut-like reset."
                ),
                "voiceover_line": (
                    f"Move from {first_analysis.mood} into {second_analysis.mood} without breaking continuity."
                ),
            }
        )

    return {"shots": shots}


def build_scene_plan(
    openai_client,
    analyses: list[ImageAnalysis],
    image_names: list[str],
    planner_model: str,
    story_brief: str,
) -> dict:
    response = openai_client.responses.create(
        model=planner_model,
        input=build_scene_director_prompt(
            analyses,
            image_names,
            story_brief,
        ),
    )
    return extract_json(response.output_text)


def build_narration_text(
    openai_client,
    analyses: list[ImageAnalysis],
    image_names: list[str],
    planner_model: str,
    story_brief: str,
) -> str:
    scene_plan = build_scene_plan(
        openai_client,
        analyses,
        image_names,
        planner_model,
        story_brief,
    )
    return str(scene_plan.get("narration_text", "")).strip()


def build_shot_blueprint(
    openai_client,
    scene_plan: dict,
    image_names: list[str],
    planner_model: str,
    max_shots: int,
    default_duration: int,
    total_duration: int,
) -> dict:
    response = openai_client.responses.create(
        model=planner_model,
        input=build_director_prompt(
            scene_plan,
            image_names,
            max_shots,
            default_duration,
            total_duration,
        ),
    )
    payload = extract_json(response.output_text)
    shots = payload.get("shots", [])
    if not shots:
        raise SystemExit("Shot planner returned no shots.")

    valid_names = set(image_names)
    for shot in shots:
        for key in ("source_image", "first_frame_image", "last_frame_image"):
            value = shot.get(key)
            if value is not None and value not in valid_names:
                raise SystemExit(f"Shot planner referenced unknown image '{value}' in field '{key}'.")
    planned_durations = allocate_shot_durations(total_duration, len(shots), default_duration)
    for shot, duration in zip(shots, planned_durations):
        shot["duration_seconds"] = duration
    return payload


def build_director_plan(
    openai_client,
    scene_plan: dict,
    shot_blueprint: dict,
    planner_model: str,
) -> dict:
    response = openai_client.responses.create(
        model=planner_model,
        input=build_prompt_director_prompt(shot_blueprint, scene_plan),
    )
    payload = extract_json(response.output_text)
    prompts_by_id = {item["shot_id"]: item for item in payload.get("shots", [])}
    merged_shots = []
    for shot in shot_blueprint.get("shots", []):
        prompt_data = prompts_by_id.get(shot["shot_id"])
        if not prompt_data:
            raise SystemExit(f"Prompt director returned no prompt for shot '{shot['shot_id']}'.")
        merged_shot = dict(shot)
        merged_shot["prompt"] = str(prompt_data["prompt"])
        merged_shot["negative_prompt"] = str(prompt_data.get("negative_prompt", ""))
        merged_shots.append(merged_shot)

    return {
        "project_title": scene_plan.get("project_title", "Untitled Project"),
        "story_arc": scene_plan.get("story_arc", ""),
        "visual_rules": scene_plan.get("visual_rules", []),
        "narration_text": scene_plan.get("narration_text", ""),
        "shots": merged_shots,
    }


def build_fallback_narration(
    analyses: list[ImageAnalysis],
    image_names: list[str],
    story_brief: str,
    total_duration: int,
) -> str:
    target_words = max(40, min(70, int(total_duration * 2.6)))
    analysis_by_name = {item.filename: item for item in analyses}
    sentences: list[str] = []

    if story_brief:
        sentences.append(story_brief.strip().rstrip(".") + ".")

    for index, name in enumerate(image_names):
        analysis = analysis_by_name.get(name)
        mood = analysis.mood if analysis and analysis.mood else "present"
        location = analysis.location if analysis and analysis.location else "a familiar place"
        subject = analysis.subjects[0] if analysis and analysis.subjects else "the scene"
        if index == 0:
            sentences.append(f"I start in {location}, feeling {mood}, watching {subject}.")
        elif index == len(image_names) - 1:
            sentences.append(f"It ends in {location}, softer and {mood} as the day settles.")
        else:
            sentences.append(f"It shifts toward {location}, still with {subject}, the mood turning {mood}.")

    sentences.extend(
        [
            "I keep the camera moving so each moment flows into the next without a hard stop.",
            "Small details carry me forward, and the rhythm stays gentle and grounded.",
            "By the end, everything feels connected, like one continuous walk through the day.",
        ]
    )

    chosen: list[str] = []
    word_count = 0
    for sentence in sentences:
        words = sentence.split()
        if chosen and word_count + len(words) > target_words:
            break
        chosen.append(sentence)
        word_count += len(words)

    if not chosen:
        return story_brief.strip()
    return " ".join(chosen)


def build_audio_output_path(audio_dir: Path) -> Path:
    audio_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return audio_dir / f"director_narration_{timestamp}.mp3"


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


def build_voice_settings_from_env() -> dict:
    def read_float(key: str, fallback: float) -> float:
        raw = os.environ.get(key)
        if raw is None or raw == "":
            return fallback
        try:
            return float(raw)
        except ValueError:
            return fallback

    def read_bool(key: str, fallback: bool) -> bool:
        raw = os.environ.get(key)
        if raw is None or raw == "":
            return fallback
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    return {
        "stability": read_float("ELEVENLABS_VOICE_STABILITY", DEFAULT_TTS_STABILITY),
        "similarity_boost": read_float(
            "ELEVENLABS_VOICE_SIMILARITY", DEFAULT_TTS_SIMILARITY_BOOST
        ),
        "style": read_float("ELEVENLABS_VOICE_STYLE", DEFAULT_TTS_STYLE),
        "use_speaker_boost": read_bool(
            "ELEVENLABS_SPEAKER_BOOST", DEFAULT_TTS_SPEAKER_BOOST
        ),
    }


def build_voice_settings(
    stability: float | None,
    similarity_boost: float | None,
    style: float | None,
    speaker_boost: bool | None,
) -> dict:
    env_settings = build_voice_settings_from_env()
    return {
        "stability": env_settings["stability"] if stability is None else float(stability),
        "similarity_boost": env_settings["similarity_boost"]
        if similarity_boost is None
        else float(similarity_boost),
        "style": env_settings["style"] if style is None else float(style),
        "use_speaker_boost": env_settings["use_speaker_boost"]
        if speaker_boost is None
        else bool(speaker_boost),
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


def generate_narration_audio(
    elevenlabs_client,
    narration_text: str,
    audio_dir: Path,
    voice_settings: dict | None,
) -> Path:
    output_path = build_audio_output_path(audio_dir)
    audio = elevenlabs_client.text_to_speech.convert(
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
        output_format="mp3_44100_128",
        text=narration_text,
        model_id=os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
        voice_settings=voice_settings,
    )
    save_audio_chunks(audio, output_path)
    return output_path


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise SystemExit(f"Expected a gs:// URI, got: {uri}")
    remainder = uri[5:]
    bucket, _, path = remainder.partition("/")
    return bucket, path


def join_gcs_path(base_uri: str, *parts: str) -> str:
    bucket, prefix = parse_gcs_uri(base_uri)
    cleaned_parts = [prefix.strip("/")] if prefix else []
    cleaned_parts.extend(part.strip("/") for part in parts if part)
    path = "/".join(part for part in cleaned_parts if part)
    return f"gs://{bucket}/{path}" if path else f"gs://{bucket}"


def upload_file_to_gcs(storage_client, local_path: Path, destination_uri: str) -> str:
    bucket_name, blob_name = parse_gcs_uri(destination_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type=mimetypes.guess_type(local_path.name)[0])
    return destination_uri


def download_file_from_gcs(storage_client, source_uri: str, destination_path: Path) -> None:
    bucket_name, blob_name = parse_gcs_uri(source_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(destination_path))


def poll_operation(genai_client, operation):
    while not operation.done:
        time.sleep(15)
        operation = genai_client.operations.get(operation)
        print("Waiting for Veo operation...", flush=True)
    return operation


def describe_operation_failure(operation) -> str:
    parts: list[str] = []

    error = getattr(operation, "error", None)
    if error is not None:
        code = getattr(error, "code", None)
        message = getattr(error, "message", None)
        if code is not None:
            parts.append(f"code={code}")
        if message:
            parts.append(f"message={message}")

    response = getattr(operation, "response", None)
    if response is not None:
        generated_videos = getattr(response, "generated_videos", None)
        if generated_videos is not None:
            parts.append(f"generated_videos={len(generated_videos)}")

    metadata = getattr(operation, "metadata", None)
    if metadata is not None:
        parts.append(f"metadata={metadata}")

    return "; ".join(parts) if parts else str(operation)


def stitch_clips(ffmpeg_path: str, clip_paths: list[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = Path(handle.name)
        for clip_path in clip_paths:
            escaped = clip_path.resolve().as_posix().replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def crossfade_clips(
    ffmpeg_path: str,
    ffprobe_path: str,
    clip_paths: list[Path],
    output_path: Path,
    crossfade_duration: float,
) -> None:
    if len(clip_paths) < 2:
        shutil.copyfile(clip_paths[0], output_path)
        return

    durations: list[float] = []
    for clip_path in clip_paths:
        probe = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(clip_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        durations.append(float(probe.stdout.strip()))

    inputs = []
    for clip_path in clip_paths:
        inputs.extend(["-i", str(clip_path)])

    filter_parts: list[str] = []
    cumulative_offset = durations[0] - crossfade_duration
    previous_video = "[0:v]"
    previous_audio = "[0:a]" if False else None
    for index in range(1, len(clip_paths)):
        next_video = f"[{index}:v]"
        output_video = f"[v{index}]"
        filter_parts.append(
            f"{previous_video}{next_video}xfade=transition=fade:duration={crossfade_duration}:offset={cumulative_offset}{output_video}"
        )
        previous_video = output_video
        cumulative_offset += durations[index] - crossfade_duration

    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            previous_video,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )


def attach_audio(ffmpeg_path: str, video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-af",
            "apad",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )


def build_shot_objects(payload: dict) -> list[ShotPlan]:
    return [
        ShotPlan(
            shot_id=str(shot["shot_id"]),
            title=str(shot["title"]),
            source_image=shot.get("source_image"),
            first_frame_image=shot.get("first_frame_image"),
            last_frame_image=shot.get("last_frame_image"),
            duration_seconds=int(shot["duration_seconds"]),
            camera=str(shot["camera"]),
            action=str(shot["action"]),
            transition_goal=str(shot["transition_goal"]),
            prompt=str(shot["prompt"]),
            negative_prompt=str(shot.get("negative_prompt", "")),
            voiceover_line=str(shot.get("voiceover_line", "")),
        )
        for shot in payload["shots"]
    ]


def resolve_anchor_name(shot: ShotPlan) -> str | None:
    return shot.first_frame_image or shot.source_image


def generate_shot_clips(
    genai_client,
    storage_client,
    shots: list[ShotPlan],
    images_by_name: dict[str, Path],
    veo_model: str,
    aspect_ratio: str,
    gcs_session_uri: str,
    local_clips_dir: Path,
) -> list[dict]:
    from google.genai.types import GenerateVideosConfig, Image

    uploaded_images: dict[str, str] = {}
    manifests: list[dict] = []

    for shot in shots:
        first_name = resolve_anchor_name(shot)
        first_image = None
        if first_name:
            if first_name not in uploaded_images:
                source_path = images_by_name[first_name]
                uploaded_images[first_name] = upload_file_to_gcs(
                    storage_client,
                    source_path,
                    join_gcs_path(gcs_session_uri, "inputs", source_path.name),
                )
            first_image = Image(
                gcs_uri=uploaded_images[first_name],
                mime_type=mimetypes.guess_type(first_name)[0] or "image/jpeg",
            )

        last_frame = None
        if shot.last_frame_image:
            if shot.last_frame_image not in uploaded_images:
                source_path = images_by_name[shot.last_frame_image]
                uploaded_images[shot.last_frame_image] = upload_file_to_gcs(
                    storage_client,
                    source_path,
                    join_gcs_path(gcs_session_uri, "inputs", source_path.name),
                )
            last_frame = Image(
                gcs_uri=uploaded_images[shot.last_frame_image],
                mime_type=mimetypes.guess_type(shot.last_frame_image)[0] or "image/jpeg",
            )

        config = GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=shot.duration_seconds,
            output_gcs_uri=join_gcs_path(gcs_session_uri, "generated", shot.shot_id),
            negative_prompt=shot.negative_prompt or None,
            generate_audio=False,
            last_frame=last_frame,
        )

        print(f"Generating {shot.shot_id}: {shot.title}", flush=True)
        if first_image is not None:
            operation = genai_client.models.generate_videos(
                model=veo_model,
                prompt=shot.prompt,
                image=first_image,
                config=config,
            )
        else:
            operation = genai_client.models.generate_videos(
                model=veo_model,
                prompt=shot.prompt,
                config=config,
            )

        operation = poll_operation(genai_client, operation)
        if not operation.response:
            raise SystemExit(
                f"Veo returned no response for {shot.shot_id}. "
                f"Operation details: {describe_operation_failure(operation)}"
            )

        generated_uri = operation.result.generated_videos[0].video.uri
        clip_path = local_clips_dir / f"{shot.shot_id}.mp4"
        download_file_from_gcs(storage_client, generated_uri, clip_path)
        manifests.append(
            {
                "shot_id": shot.shot_id,
                "title": shot.title,
                "local_clip_path": str(clip_path),
                "gcs_video_uri": generated_uri,
                "prompt": shot.prompt,
            }
        )
        print(f"Saved clip to {clip_path}", flush=True)

    return manifests


def main() -> None:
    load_local_env()
    args = parse_args()

    require_env("OPENAI_API_KEY")
    if not args.skip_video:
        require_env("GOOGLE_CLOUD_PROJECT")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

    if not args.gcs_output_uri and not args.skip_video:
        raise SystemExit(
            "Missing GCS output prefix. Set GOOGLE_CLOUD_VEO_OUTPUT_GCS_URI or pass --gcs-output-uri."
        )

    try:
        from google import genai
        from google.cloud import storage
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install requirements with `pip install -r requirements.txt`."
        ) from exc

    ffmpeg_path = None
    ffprobe_path = None
    if not args.skip_video:
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if not ffmpeg_path:
            raise SystemExit("ffmpeg must be installed and available on PATH.")
        if args.soft_transitions and not ffprobe_path:
            raise SystemExit("`--soft-transitions` requires ffprobe to be installed and available on PATH.")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    image_paths = collect_images(input_dir)
    images_by_name = {path.name: path for path in image_paths}
    effective_max_shots = min(args.max_shots, max_shots_for_total_duration(args.total_duration))
    if effective_max_shots < 1:
        raise SystemExit("Total duration is too short for Veo. Minimum supported final length is 4 seconds.")

    run_id = datetime.now().strftime("director_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    clips_dir = run_dir / "clips"
    run_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    openai_client = OpenAI()
    analyses = analyze_images(openai_client, image_paths, args.image_model)
    analyses_path = run_dir / "image_analysis.json"
    save_json(analyses_path, [asdict(item) for item in analyses])

    image_names = [path.name for path in image_paths]
    if args.sequence_mode == "adjacent-pairs":
        if len(image_names) < 2:
            raise SystemExit("`adjacent-pairs` mode requires at least 2 images.")
        scene_plan = build_adjacent_pair_scene_plan(analyses, image_names, args.story_brief)
        model_narration = build_narration_text(
            openai_client,
            analyses,
            image_names,
            args.planner_model,
            args.story_brief,
        )
        if model_narration:
            scene_plan["narration_text"] = model_narration
    else:
        scene_plan = build_scene_plan(
            openai_client,
            analyses,
            image_names,
            args.planner_model,
            args.story_brief,
        )
    save_json(run_dir / "scene_plan.json", scene_plan)

    if args.sequence_mode == "adjacent-pairs":
        shot_blueprint = build_adjacent_pair_shot_blueprint(
            analyses,
            image_names,
            args.total_duration,
            args.default_shot_duration,
        )
    else:
        shot_blueprint = build_shot_blueprint(
            openai_client,
            scene_plan,
            image_names,
            args.planner_model,
            effective_max_shots,
            args.default_shot_duration,
            args.total_duration,
        )
    save_json(run_dir / "shot_blueprint.json", shot_blueprint)

    director_plan = build_director_plan(
        openai_client,
        scene_plan,
        shot_blueprint,
        args.planner_model,
    )
    if not str(director_plan.get("narration_text", "")).strip():
        director_plan["narration_text"] = build_fallback_narration(
            analyses,
            image_names,
            args.story_brief,
            args.total_duration,
        )
    plan_path = run_dir / "director_plan.json"
    save_json(plan_path, director_plan)
    shots = build_shot_objects(director_plan)
    print(
        f"Built director plan with {len(shots)} shot(s), target total {args.total_duration}s, "
        f"actual planned total {sum(shot.duration_seconds for shot in shots)}s",
        flush=True,
    )

    formatted_narration = format_narration_for_tts(director_plan["narration_text"])
    narration_path = run_dir / "narration.txt"
    narration_path.write_text(formatted_narration.strip() + "\n", encoding="utf-8")

    audio_path = None
    if not args.skip_audio:
        try:
            from elevenlabs import ElevenLabs
        except ImportError as exc:
            raise SystemExit(
                "Audio generation requires the optional `elevenlabs` package. "
                "Install it separately or run with `--skip-audio`."
            ) from exc
        require_env("ELEVENLABS_API_KEY")
        elevenlabs_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
        voice_settings = build_voice_settings(
            args.tts_stability,
            args.tts_similarity_boost,
            args.tts_style,
            args.tts_speaker_boost,
        )
        audio_path = generate_narration_audio(
            elevenlabs_client,
            formatted_narration,
            audio_dir,
            voice_settings,
        )
        print(f"Saved narration audio to {audio_path}", flush=True)

    clip_manifest: list[dict] = []
    if not args.skip_video:
        genai_client = genai.Client()
        storage_client = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"])
        gcs_session_uri = join_gcs_path(args.gcs_output_uri, run_id)
        clip_manifest = generate_shot_clips(
            genai_client,
            storage_client,
            shots,
            images_by_name,
            args.veo_model,
            args.aspect_ratio,
            gcs_session_uri,
            clips_dir,
        )
        save_json(run_dir / "clip_manifest.json", clip_manifest)

        stitched_path = run_dir / "final_video.mp4"
        clip_paths = [Path(item["local_clip_path"]) for item in clip_manifest]
        if args.soft_transitions and len(clip_paths) > 1:
            crossfade_clips(
                ffmpeg_path,
                ffprobe_path,
                clip_paths,
                stitched_path,
                args.crossfade_duration,
            )
        else:
            stitch_clips(
                ffmpeg_path,
                clip_paths,
                stitched_path,
            )
        print(f"Saved stitched video to {stitched_path}", flush=True)

        if audio_path is not None:
            final_with_audio = run_dir / "final_video_with_audio.mp4"
            attach_audio(ffmpeg_path, stitched_path, audio_path, final_with_audio)
            print(f"Saved final video with audio to {final_with_audio}", flush=True)

    print(f"Director pipeline artifacts saved in {run_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Cancelled by user.")
