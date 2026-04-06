#!/usr/bin/env python3

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from analyze_image import SUPPORTED_SUFFIXES


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BASE_DIR / "web_uploads"
SAMPLE_INPUT_DIR = BASE_DIR / "input_images"
VIDEO_ROOT = BASE_DIR / "generated_videos"
DEFAULT_OUTPUT_FILE = BASE_DIR / "descriptions.txt"
DEFAULT_AUDIO_DIR = BASE_DIR / "audio files"
MAX_UPLOAD_SIZE_BYTES = 32 * 1024 * 1024

# Ensure required directories exist in fresh environments (e.g., Render).
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_BYTES


def is_supported_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_SUFFIXES


def ensure_project_child(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(BASE_DIR)
    except ValueError as exc:
        raise FileNotFoundError("Requested file is outside the project directory.") from exc
    return resolved


def ensure_audio_child(path: Path) -> Path:
    resolved = ensure_project_child(path)
    try:
        resolved.relative_to(DEFAULT_AUDIO_DIR.resolve())
    except ValueError as exc:
        raise FileNotFoundError("Requested file is outside the audio directory.") from exc
    return resolved


def ensure_video_child(path: Path) -> Path:
    resolved = ensure_project_child(path)
    try:
        resolved.relative_to(VIDEO_ROOT.resolve())
    except ValueError as exc:
        raise FileNotFoundError("Requested file is outside the generated videos directory.") from exc
    return resolved


def list_sample_images(limit: int = 4) -> list[Path]:
    if not SAMPLE_INPUT_DIR.is_dir():
        return []
    images = [
        path
        for path in sorted(SAMPLE_INPUT_DIR.iterdir())
        if path.is_file() and is_supported_image(path.name)
    ]
    return images[:limit]


def save_uploaded_images(files, required_count: int = 4) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = UPLOAD_ROOT / f"run_{timestamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for index, file_storage in enumerate(files, start=1):
        original_name = file_storage.filename or f"image_{index}.jpg"
        if not is_supported_image(original_name):
            raise ValueError(
                f"Unsupported file type for {original_name}. Please upload PNG, JPG, JPEG, WEBP, or GIF images."
            )

        sanitized_name = secure_filename(original_name) or f"image_{index}{Path(original_name).suffix.lower()}"
        final_name = f"{index:02d}_{sanitized_name}"
        destination = run_dir / final_name
        file_storage.save(destination)
        saved_paths.append(destination)

    if not saved_paths:
        raise ValueError("Please upload at least one image.")
    if len(saved_paths) < required_count:
        raise ValueError(f"Please upload {required_count} images.")

    return saved_paths


def run_director_pipeline(
    input_dir: Path,
    total_duration: int = 20,
    generate_video: bool = False,
    sequence_mode: str = "adjacent-pairs",
) -> dict:
    cmd = [
        sys.executable,
        str(BASE_DIR / "director_pipeline.py"),
        "--input-dir",
        str(input_dir),
        "--total-duration",
        str(total_duration),
        "--sequence-mode",
        sequence_mode,
    ]
    if not generate_video:
        cmd.append("--skip-video")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        details = "\n".join(line for line in [stdout.strip(), stderr.strip()] if line)
        raise RuntimeError(f"Pipeline failed.\n{details}" if details else "Pipeline failed.") from exc
    run_dirs = [path for path in VIDEO_ROOT.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError("No generated video runs were found.")
    latest_run = max(run_dirs, key=lambda path: path.stat().st_mtime)
    narration_path = latest_run / "narration.txt"
    narration_text = narration_path.read_text(encoding="utf-8") if narration_path.is_file() else ""
    analysis_path = latest_run / "image_analysis.json"
    scene_plan_path = latest_run / "scene_plan.json"
    shot_blueprint_path = latest_run / "shot_blueprint.json"
    director_plan_path = latest_run / "director_plan.json"
    audio_candidates = sorted(
        DEFAULT_AUDIO_DIR.glob("director_narration_*.mp3"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    audio_path = audio_candidates[0] if audio_candidates else None
    final_with_audio = latest_run / "final_video_with_audio.mp4"
    final_video = latest_run / "final_video.mp4"
    video_path = final_with_audio if final_with_audio.is_file() else final_video if final_video.is_file() else None
    return {
        "run_dir": latest_run,
        "narration_text": narration_text.strip(),
        "analysis_path": analysis_path if analysis_path.is_file() else None,
        "scene_plan_path": scene_plan_path if scene_plan_path.is_file() else None,
        "shot_blueprint_path": shot_blueprint_path if shot_blueprint_path.is_file() else None,
        "director_plan_path": director_plan_path if director_plan_path.is_file() else None,
        "audio_path": audio_path,
        "video_path": video_path,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "error": None,
        "result": None,
        "sample_images": [
            {
                "label": f"Image {index}",
                "name": path.name,
                "url": url_for("serve_sample_image", filename=path.name),
            }
            for index, path in enumerate(list_sample_images(), start=1)
        ],
    }

    if request.method == "POST":
        uploaded_files = [
            request.files.get("image_1"),
            request.files.get("image_2"),
            request.files.get("image_3"),
            request.files.get("image_4"),
        ]
        uploaded_files = [file for file in uploaded_files if file and file.filename]
        use_samples = request.form.get("use_samples") == "on"
        generate_video = request.form.get("generate_video") == "on"
        sequence_mode = request.form.get("sequence_mode", "adjacent-pairs")

        if not uploaded_files and not use_samples:
            context["error"] = "Choose one or more images or select the sample set."
            return render_template("index.html", **context), 400

        try:
            if use_samples:
                saved_paths = list_sample_images()
                if not saved_paths:
                    raise ValueError("No sample images were found in input_images.")
                input_dir = SAMPLE_INPUT_DIR
            else:
                saved_paths = save_uploaded_images(uploaded_files, required_count=4)
                input_dir = saved_paths[0].parent
            director_result = run_director_pipeline(
                input_dir=input_dir,
                total_duration=20,
                generate_video=generate_video,
                sequence_mode=sequence_mode,
            )
            audio_relative = None
            audio_url = None
            audio_path = director_result["audio_path"]
            if audio_path:
                audio_relative = audio_path.resolve().relative_to(DEFAULT_AUDIO_DIR.resolve()).as_posix()
                audio_url = url_for("serve_audio_file", relative_path=audio_relative)
            video_path = director_result["video_path"]
            video_url = None
            if video_path:
                video_relative = video_path.resolve().relative_to(VIDEO_ROOT.resolve()).as_posix()
                video_url = url_for("serve_video_file", relative_path=video_relative)
            run_dir = director_result["run_dir"]
            def rel_path(path: Path | None) -> str | None:
                if not path:
                    return None
                return path.resolve().relative_to(BASE_DIR).as_posix()

            def read_text_file(path: Path | None) -> str | None:
                if not path or not path.is_file():
                    return None
                try:
                    return path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    return path.read_text(encoding="utf-8", errors="replace")

            analysis_relative = rel_path(director_result["analysis_path"])
            scene_relative = rel_path(director_result["scene_plan_path"])
            blueprint_relative = rel_path(director_result["shot_blueprint_path"])
            plan_relative = rel_path(director_result["director_plan_path"])
            analysis_text = read_text_file(director_result["analysis_path"])
            scene_text = read_text_file(director_result["scene_plan_path"])
            blueprint_text = read_text_file(director_result["shot_blueprint_path"])
            plan_text = read_text_file(director_result["director_plan_path"])
            narration_text = director_result["narration_text"] or "[No narration returned]"
            context["result"] = {
                "image_count": len(saved_paths),
                "image_names": [path.name for path in saved_paths],
                "run_text": narration_text,
                "audio_url": audio_url,
                "audio_path": audio_path.resolve().relative_to(BASE_DIR).as_posix()
                if audio_path
                else None,
                "descriptions_path": plan_relative,
                "analysis_path": analysis_relative,
                "analysis_text": analysis_text,
                "scene_plan_path": scene_relative,
                "scene_plan_text": scene_text,
                "shot_blueprint_path": blueprint_relative,
                "shot_blueprint_text": blueprint_text,
                "director_plan_path": plan_relative,
                "director_plan_text": plan_text,
                "video_url": video_url,
                "video_path": video_path.resolve().relative_to(BASE_DIR).as_posix() if video_path else None,
            }
        except ValueError as exc:
            context["error"] = str(exc)
            return render_template("index.html", **context), 400
        except SystemExit as exc:
            context["error"] = str(exc)
            return render_template("index.html", **context), 500
        except Exception as exc:
            context["error"] = str(exc)
            return render_template("index.html", **context), 500

    return render_template("index.html", **context)


@app.route("/audio/<path:relative_path>")
def serve_audio_file(relative_path: str):
    target_path = ensure_audio_child(DEFAULT_AUDIO_DIR / relative_path)
    if not target_path.is_file():
        abort(404)
    return send_file(target_path)


@app.route("/video/<path:relative_path>")
def serve_video_file(relative_path: str):
    target_path = ensure_video_child(VIDEO_ROOT / relative_path)
    if not target_path.is_file():
        abort(404)
    return send_file(target_path)


@app.route("/sample/<path:filename>")
def serve_sample_image(filename: str):
    target_path = ensure_project_child(SAMPLE_INPUT_DIR / filename)
    if not target_path.is_file():
        abort(404)
    return send_file(target_path)


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(_error):
    return (
        render_template(
            "index.html",
            error="Upload too large. Keep the total request under 32 MB.",
            result=None,
        ),
        413,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(debug=False, host="0.0.0.0", port=port)
