"""List model files on the worker's filesystem.

Scans both the baked-in ComfyUI models directory and the network volume
models directory for files matching a given model type (e.g. loras).

Also reads extra_model_paths.yaml if present for additional search paths.
"""

import os

try:
    import yaml
except ImportError:
    yaml = None

COMFYUI_MODELS = "/ComfyUI/models"
VOLUME_MODELS = "/runpod-volume/ComfyUI/models"
EXTRA_PATHS_FILE = "/ComfyUI/extra_model_paths.yaml"

# File extensions we care about
MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf"}


def _read_extra_paths(model_type: str) -> list[str]:
    """Read extra_model_paths.yaml and return additional directories for a model type."""
    if not os.path.isfile(EXTRA_PATHS_FILE) or yaml is None:
        return []

    try:
        with open(EXTRA_PATHS_FILE) as f:
            data = yaml.safe_load(f)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    paths = []
    for _section_name, section in data.items():
        if not isinstance(section, dict):
            continue
        base_path = section.get("base_path", "")
        subfolder = section.get(model_type, "")
        if base_path and subfolder:
            full_path = os.path.join(base_path, subfolder)
            if full_path not in paths:
                paths.append(full_path)

    return paths


def _list_files(directory: str) -> list[dict]:
    """List model files under a directory, recursing into subfolders.

    ComfyUI references models in subfolders by their root-relative path
    (e.g. "folder_a/x.safetensors"), so filename carries that relative path
    with forward slashes to match what the workflow's lora_name expects.
    """
    if not os.path.isdir(directory):
        return []

    files = []
    for root, _dirs, names in os.walk(directory):
        for name in names:
            _, ext = os.path.splitext(name)
            if ext.lower() not in MODEL_EXTENSIONS:
                continue
            filepath = os.path.join(root, name)
            if not os.path.isfile(filepath):
                continue
            rel = os.path.relpath(filepath, directory).replace(os.sep, "/")
            size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)
            files.append({
                "filename": rel,
                "path": filepath,
                "size_mb": size_mb,
            })

    return files


def handle(job: dict) -> dict:
    """Handle a list_models command.

    Expected input:
    {
        "command": "list_models",
        "model_type": "loras"
    }

    Returns:
    {
        "ok": true,
        "model_type": "loras",
        "files": [
            {"filename": "my_lora.safetensors", "path": "/runpod-volume/...", "size_mb": 228.5}
        ],
        "search_paths": ["/ComfyUI/models/loras", "/runpod-volume/ComfyUI/models/loras"]
    }
    """
    job_input = job["input"]
    model_type = job_input.get("model_type", "loras")

    # Build search paths
    search_paths = [
        os.path.join(COMFYUI_MODELS, model_type),
        os.path.join(VOLUME_MODELS, model_type),
    ]

    # Add paths from extra_model_paths.yaml
    extra = _read_extra_paths(model_type)
    for p in extra:
        if p not in search_paths:
            search_paths.append(p)

    # Collect files, deduplicating by filename (volume takes precedence)
    seen = {}
    for directory in search_paths:
        for f in _list_files(directory):
            if f["filename"] not in seen:
                seen[f["filename"]] = f

    files = sorted(seen.values(), key=lambda f: f["filename"])

    return {
        "ok": True,
        "model_type": model_type,
        "files": files,
        "search_paths": [p for p in search_paths if os.path.isdir(p)],
    }
