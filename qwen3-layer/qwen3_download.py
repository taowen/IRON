"""Download the MyLM Qwen3-8B Q4NX model assets."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, MODEL_FILES, missing_model_files, model_file_url


def download_qwen3_8b_model(model_path: Path = DEFAULT_QWEN3_8B_MODEL_PATH) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES:
        destination = model_path / filename
        if destination.exists():
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        url = model_file_url(filename)
        print(f"downloading {filename} -> {destination}")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)


def ensure_qwen3_8b_model(model_path: Path = DEFAULT_QWEN3_8B_MODEL_PATH, download: bool = False) -> None:
    missing = missing_model_files(model_path)
    if not missing:
        return
    if not download:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Qwen3-8B-NPU2 model files are missing. "
            "Run with --download-model to fetch them:\n"
            f"{formatted}"
        )
    download_qwen3_8b_model(model_path)
