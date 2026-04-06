#!/usr/bin/env python3

import argparse
import base64
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL = "gen4_turbo"
DEFAULT_TOTAL_DURATION = 7
DEFAULT_RATIO = "1280:720"
DEFAULT_GROUP_SIZE = 2
DEFAULT_INPUT_DIR = "input_images"
DEFAULT_DESCRIPTIONS_FILE = "descriptions.txt"
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_ALLOWED_DURATIONS = {
    "gen4_turbo": (5, 10),
    "gen4.5": tuple(range(2, 11)),
    "gen3a_turbo": (5, 10),
    "veo3": (8,),
    "veo3.1": (4, 6, 8),
    "veo3.1_fast": (4, 6, 8),
}
MODEL_SUPPORTS_LAST_FRAME = {
    "gen4_turbo": False,
    "gen4.5": False,
    "gen3a_turbo": True,
    "veo3": False,
    "veo3.1": True,
    "veo3.1_fast": True,
}
DEFAULT_FALLBACK_PROMPT = (
    "Create a smooth cinematic travel journal video using two input images. "
    "Start from the first image as a hotel scene. Introduce a traveler naturally in the environment, "
    "preparing to leave, stepping outside, or beginning to walk forward. Add subtle environmental motion such as wind, "
    "soft lighting shifts, fabric movement, and gentle atmosphere changes. "
    "The traveler begins moving forward and the camera follows with a slow, steady cinematic tracking motion. "
    "Seamlessly transform the environment into one continuous journey with no cuts, no slideshow transitions, and no abrupt changes. "
    "As the traveler moves, let the streets, light, and atmosphere evolve naturally into Parliament Hill in Ottawa. "
    "In the final moment, the same traveler continues walking or pauses to take in the view. "
    "Maintain realism, continuous camera motion, natural light, moving clouds, waving flags, and an immersive story-driven travel diary feeling. "
    "Do not use cuts, slides, freeze frames, or PPT-style transitions."
)
MAX_DATA_URI_BYTES = 5 * 1024 * 1024
MAX_TASK_RETRIES = 3
RETRY_DELAY_SECONDS = 10
TEMP_IMAGE_QUALITY_STEPS = (90, 80, 70, 60, 50, 40)
TEMP_IMAGE_MAX_DIMENSIONS = (1920, 1920)

# By default, video generation uses the latest Narrative block from descriptions.txt.
# This fallback prompt is only used if that file is missing or contains no narrative.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Runway image-to-video clips from a folder of images. "
            "Each batch of images becomes transition clips between adjacent pairs, with optional stitching "
            "into one final video."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing source images. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_videos",
        help="Folder where clips and merged videos will be saved. Default: generated_videos",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"How many images belong to one final video. Default: {DEFAULT_GROUP_SIZE}",
    )
    parser.add_argument(
        "--total-duration",
        type=int,
        default=DEFAULT_TOTAL_DURATION,
        help=(
            "Target total duration in seconds for each final batch video. "
            f"Default: {DEFAULT_TOTAL_DURATION}"
        ),
    )
    parser.add_argument(
        "--ratio",
        default=DEFAULT_RATIO,
        help=f"Runway output ratio. Default: {DEFAULT_RATIO}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Runway model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--descriptions-file",
        default=DEFAULT_DESCRIPTIONS_FILE,
        help=(
            "Text file containing generated image descriptions and narratives. "
            f"Default: {DEFAULT_DESCRIPTIONS_FILE}"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Prompt text applied to every transition clip. "
            "If omitted, the latest Narrative from the descriptions file is used."
        ),
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Merge all generated clips in a group into one final MP4 using ffmpeg.",
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Also process a final partial batch if the image count is not divisible by the group size.",
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


def require_api_key() -> None:
    if not os.environ.get("RUNWAYML_API_SECRET"):
        raise SystemExit(
            "Missing RUNWAYML_API_SECRET. Add it to your shell or a local .env file before running this script."
        )


def load_latest_narrative(descriptions_path: Path) -> str:
    if not descriptions_path.is_file():
        return DEFAULT_FALLBACK_PROMPT

    content = descriptions_path.read_text(encoding="utf-8")
    marker = "Narrative:"
    marker_index = content.rfind(marker)
    if marker_index == -1:
        return DEFAULT_FALLBACK_PROMPT

    narrative = content[marker_index + len(marker) :].strip()
    if not narrative:
        return DEFAULT_FALLBACK_PROMPT

    lines: list[str] = []
    for line in narrative.splitlines():
        stripped = line.strip()
        if stripped.startswith("Run: "):
            break
        lines.append(stripped)

    cleaned = " ".join(part for part in lines if part).strip()
    return cleaned or DEFAULT_FALLBACK_PROMPT


def resolve_prompt_text(args: argparse.Namespace, descriptions_path: Path) -> tuple[str, str]:
    if args.prompt:
        return args.prompt, "manual --prompt"

    narrative = load_latest_narrative(descriptions_path)
    if narrative != DEFAULT_FALLBACK_PROMPT:
        return narrative, f"latest Narrative from {descriptions_path.name}"

    return narrative, "fallback built-in prompt"


def list_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    preferred_images = [input_dir / "image1.jpg", input_dir / "image2.png"]
    if all(path.is_file() for path in preferred_images):
        return preferred_images

    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS
    )

    if not images:
        raise SystemExit(f"No supported images found in: {input_dir}")

    return images


def chunk_images(images: list[Path], group_size: int, keep_incomplete: bool) -> list[list[Path]]:
    groups: list[list[Path]] = []
    for index in range(0, len(images), group_size):
        group = images[index : index + group_size]
        if len(group) < group_size and not keep_incomplete:
            break
        if len(group) >= 2:
            groups.append(group)
    return groups


def split_total_duration(total_duration: int, num_clips: int) -> list[int]:
    if num_clips < 1:
        return []

    if total_duration < num_clips:
        raise SystemExit(
            f"--total-duration must be at least {num_clips} second(s) when generating {num_clips} clip(s)."
        )

    base_duration = total_duration // num_clips
    remainder = total_duration % num_clips
    durations = [base_duration] * num_clips

    for index in range(remainder):
        durations[index] += 1

    return durations


def plan_clip_durations(total_duration: int, num_clips: int, model: str) -> list[int]:
    if num_clips < 1:
        return []

    requested = split_total_duration(max(total_duration, num_clips), num_clips)
    allowed_durations = MODEL_ALLOWED_DURATIONS.get(model)

    if not allowed_durations:
        return requested

    planned: list[int] = []
    for duration in requested:
        selected = allowed_durations[-1]
        for allowed in allowed_durations:
            if duration <= allowed:
                selected = allowed
                break
        planned.append(selected)

    return planned


def build_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {image_path.name}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def create_small_temp_image(image_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Image optimization requires Pillow. Install it with `pip install -r requirements.txt`."
        ) from exc

    with Image.open(image_path) as original_image:
        image = original_image.convert("RGB")
        image.thumbnail(TEMP_IMAGE_MAX_DIMENSIONS)

        for quality in TEMP_IMAGE_QUALITY_STEPS:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                temp_path = Path(handle.name)

            image.save(temp_path, format="JPEG", quality=quality, optimize=True)
            if temp_path.stat().st_size <= MAX_DATA_URI_BYTES:
                return temp_path

            temp_path.unlink(missing_ok=True)

    raise SystemExit(
        f"Could not compress {image_path.name} below 5 MB for upload. "
        "Try a smaller source image."
    )


def resolve_prompt_image(client, image_path: Path) -> str:
    file_size = image_path.stat().st_size
    uploads_api = getattr(client, "uploads", None)

    if uploads_api and hasattr(uploads_api, "create_ephemeral"):
        try:
            with image_path.open("rb") as image_file:
                upload = uploads_api.create_ephemeral(file=image_file)
            upload_uri = getattr(upload, "uri", None)
            if upload_uri:
                return upload_uri
        except Exception as exc:
            error_name = exc.__class__.__name__
            if error_name != "PermissionDeniedError":
                raise

    if file_size <= MAX_DATA_URI_BYTES:
        return build_data_uri(image_path)

    print(
        f"{image_path.name} is larger than 5 MB. Creating a temporary optimized copy for upload...",
        flush=True,
    )
    temp_image_path = create_small_temp_image(image_path)
    try:
        return build_data_uri(temp_image_path)
    finally:
        temp_image_path.unlink(missing_ok=True)


def build_prompt_image_payload(model: str, first_ref: str, last_ref: str | None) -> str | list[dict[str, str]]:
    if MODEL_SUPPORTS_LAST_FRAME.get(model, False) and last_ref:
        return [
            {"uri": first_ref, "position": "first"},
            {"uri": last_ref, "position": "last"},
        ]

    return first_ref


def extract_video_url(task_output) -> str:
    if isinstance(task_output, str):
        return task_output

    if isinstance(task_output, dict):
        for key in ("url", "video_url"):
            if task_output.get(key):
                return task_output[key]
        if isinstance(task_output.get("output"), list) and task_output["output"]:
            first_item = task_output["output"][0]
            if isinstance(first_item, str):
                return first_item
            if isinstance(first_item, dict):
                return first_item.get("url") or first_item.get("video_url")

    output_list = getattr(task_output, "output", None)
    if isinstance(output_list, list) and output_list:
        first_item = output_list[0]
        if isinstance(first_item, str):
            return first_item
        if isinstance(first_item, dict):
            return first_item.get("url") or first_item.get("video_url")

    for attr in ("url", "video_url"):
        value = getattr(task_output, attr, None)
        if value:
            return value

    raise SystemExit("Runway task completed, but no downloadable video URL was found in the response.")


def sanitize_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value).strip("_")


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)


def should_retry_task_failure(exc) -> bool:
    failure_code = getattr(exc, "task_details", None)
    if failure_code is None:
        return False

    code = getattr(failure_code, "failure_code", "")
    message = getattr(failure_code, "failure", "")
    normalized = f"{code} {message}".upper()
    return "INTERNAL" in normalized or "HIGH LOAD" in normalized


def stitch_clips(ffmpeg_path: str, clip_paths: Iterable[Path], output_path: Path) -> None:
    clip_paths = list(clip_paths)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = Path(handle.name)
        for clip in clip_paths:
            escaped = clip.resolve().as_posix().replace("'", "'\\''")
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


def main() -> None:
    args = parse_args()
    load_local_env()
    require_api_key()

    try:
        from runwayml import RunwayML, TaskFailedError
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: runwayml. Install it with `pip install -r requirements.txt`."
        ) from exc

    if args.group_size < 2:
        raise SystemExit("--group-size must be at least 2.")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    descriptions_path = Path(args.descriptions_file).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_text, prompt_source = resolve_prompt_text(args, descriptions_path)

    ffmpeg_path = shutil.which("ffmpeg")
    if args.stitch and not ffmpeg_path:
        raise SystemExit("`--stitch` requires ffmpeg to be installed and available on PATH.")

    images = list_images(input_dir)
    groups = chunk_images(images, args.group_size, args.keep_incomplete)

    if not groups:
        raise SystemExit("Not enough images to create a video batch.")

    client = RunwayML()

    print(
        f"Found {len(images)} images. Creating {len(groups)} batch(es) with up to {args.group_size} images each.",
        flush=True,
    )
    print(
        f"Using {args.model} to build a multi-clip transition sequence from adjacent image pairs.",
        flush=True,
    )
    print(f"Prompt source: {prompt_source}", flush=True)
    print(f"Using prompt: {prompt_text}", flush=True)

    for group_index, group in enumerate(groups, start=1):
        group_name = f"group_{group_index:02d}"
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {group_name} with {len(group)} image(s)...", flush=True)

        clip_paths: list[Path] = []
        clip_durations = plan_clip_durations(args.total_duration, len(group) - 1, args.model)

        actual_total_duration = sum(clip_durations)
        if actual_total_duration != args.total_duration:
            print(
                f"Adjusted total duration from {args.total_duration}s to {actual_total_duration}s "
                f"to fit valid clip lengths for model {args.model}.",
                flush=True,
            )

        for clip_index, ((first_image, last_image), clip_duration) in enumerate(
            zip(zip(group, group[1:]), clip_durations), start=1
        ):
            first_ref = resolve_prompt_image(client, first_image)
            last_ref = resolve_prompt_image(client, last_image)
            prompt_image_payload = build_prompt_image_payload(args.model, first_ref, last_ref)

            clip_name = (
                f"{group_name}_clip_{clip_index:02d}_"
                f"{sanitize_name(first_image.stem)}_to_{sanitize_name(last_image.stem)}.mp4"
            )
            clip_path = group_dir / clip_name

            print(
                f"Generating clip {clip_index}/{len(group) - 1}: "
                f"{first_image.name} -> {last_image.name} ({clip_duration}s)",
                flush=True,
            )
            if not MODEL_SUPPORTS_LAST_FRAME.get(args.model, False):
                print(
                    f"Model {args.model} only supports a first-frame prompt image. "
                    f"{last_image.name} will guide the narrative indirectly but cannot be sent as a true last frame.",
                    flush=True,
                )

            for attempt in range(1, MAX_TASK_RETRIES + 1):
                try:
                    task = client.image_to_video.create(
                        model=args.model,
                        prompt_image=prompt_image_payload,
                        prompt_text=prompt_text,
                        ratio=args.ratio,
                        duration=clip_duration,
                        audio=False,
                    ).wait_for_task_output()
                    break
                except TaskFailedError as exc:
                    if attempt == MAX_TASK_RETRIES or not should_retry_task_failure(exc):
                        raise SystemExit(
                            f"Runway failed on {first_image.name} -> {last_image.name}: {exc.task_details}"
                        ) from exc

                    print(
                        f"Runway is under high load for {first_image.name} -> {last_image.name}. "
                        f"Retrying in {RETRY_DELAY_SECONDS}s ({attempt}/{MAX_TASK_RETRIES})...",
                        flush=True,
                    )
                    time.sleep(RETRY_DELAY_SECONDS)

            video_url = extract_video_url(task)
            download_file(video_url, clip_path)
            clip_paths.append(clip_path)
            print(f"Saved clip to {clip_path}", flush=True)

        if args.stitch and clip_paths:
            merged_path = group_dir / f"{group_name}_final.mp4"
            print(f"Stitching {len(clip_paths)} clips into {merged_path.name}", flush=True)
            stitch_clips(ffmpeg_path, clip_paths, merged_path)
            print(f"Saved merged video to {merged_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Cancelled by user.")
