"""Image shrink recovery for provider ``image_too_large`` errors."""

from __future__ import annotations

import base64
import copy
import math
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any


DEFAULT_MAX_BASE64_BYTES = 5 * 1024 * 1024
DEFAULT_TARGET_BASE64_BYTES = 4 * 1024 * 1024


@dataclass(slots=True)
class ImageShrinkStats:
    candidates: int = 0
    changed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    original_base64_bytes_max: int = 0
    shrunk_base64_bytes_max: int = 0
    max_base64_bytes: int = DEFAULT_MAX_BASE64_BYTES
    target_base64_bytes: int = DEFAULT_TARGET_BASE64_BYTES
    error_reasons: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.changed_count > 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "changed_count": self.changed_count,
            "candidates": self.candidates,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "original_base64_bytes_max": self.original_base64_bytes_max,
            "shrunk_base64_bytes_max": self.shrunk_base64_bytes_max,
            "max_base64_bytes": self.max_base64_bytes,
            "target_base64_bytes": self.target_base64_bytes,
            "error_reasons": list(self.error_reasons),
        }


def shrink_image_parts_in_messages(
    messages: list[dict[str, Any]],
    *,
    max_base64_bytes: int = DEFAULT_MAX_BASE64_BYTES,
    target_base64_bytes: int = DEFAULT_TARGET_BASE64_BYTES,
) -> tuple[list[dict[str, Any]], ImageShrinkStats]:
    """Return a deep-copied message list with oversized base64 images shrunk."""

    copied = copy.deepcopy(messages)
    stats = ImageShrinkStats(
        max_base64_bytes=max(1, int(max_base64_bytes)),
        target_base64_bytes=max(1, int(target_base64_bytes)),
    )
    _shrink_value(copied, stats=stats, seen=set())
    return copied, stats


def _shrink_value(value: Any, *, stats: ImageShrinkStats, seen: set[int]) -> None:
    if isinstance(value, (str, bytes, bytearray, memoryview)) or value is None:
        return

    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, list):
        for item in value:
            _shrink_value(item, stats=stats, seen=seen)
        return

    if not isinstance(value, dict):
        return

    if _try_shrink_openai_chat_image(value, stats=stats):
        return
    if _try_shrink_openai_responses_image(value, stats=stats):
        return
    if _try_shrink_anthropic_image(value, stats=stats):
        return

    for item in value.values():
        _shrink_value(item, stats=stats, seen=seen)


def _try_shrink_openai_chat_image(block: dict[str, Any], *, stats: ImageShrinkStats) -> bool:
    if block.get("type") != "image_url":
        return False
    image_url = block.get("image_url")
    if not isinstance(image_url, dict):
        return False
    url = image_url.get("url")
    if not isinstance(url, str):
        return False
    updated = _shrink_data_url(url, stats=stats)
    if updated is not None:
        image_url["url"] = updated
    return True


def _try_shrink_openai_responses_image(block: dict[str, Any], *, stats: ImageShrinkStats) -> bool:
    if block.get("type") != "input_image":
        return False
    image_url = block.get("image_url")
    if not isinstance(image_url, str):
        return False
    updated = _shrink_data_url(image_url, stats=stats)
    if updated is not None:
        block["image_url"] = updated
    return True


def _try_shrink_anthropic_image(block: dict[str, Any], *, stats: ImageShrinkStats) -> bool:
    if block.get("type") != "image":
        return False
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        return False
    data = source.get("data")
    media_type = source.get("media_type")
    if not isinstance(data, str) or not isinstance(media_type, str):
        return False
    updated = _shrink_base64_payload(data, media_type=media_type, stats=stats)
    if updated is not None:
        updated_base64, updated_media_type = updated
        source["data"] = updated_base64
        source["media_type"] = updated_media_type
    return True


def _shrink_data_url(url: str, *, stats: ImageShrinkStats) -> str | None:
    parsed = _parse_data_image_url(url)
    if parsed is None:
        return None
    prefix, media_type, data = parsed
    updated = _shrink_base64_payload(data, media_type=media_type, stats=stats)
    if updated is None:
        return None
    updated_base64, updated_media_type = updated
    if updated_media_type == media_type:
        return f"{prefix},{updated_base64}"
    return f"data:{updated_media_type};base64,{updated_base64}"


def _parse_data_image_url(url: str) -> tuple[str, str, str] | None:
    if not url.startswith("data:image/") or ";base64," not in url:
        return None
    prefix, data = url.split(",", 1)
    header = prefix.removeprefix("data:")
    media_type = header.split(";", 1)[0].strip().lower()
    if not media_type.startswith("image/") or not data:
        return None
    return prefix, media_type, data


def _shrink_base64_payload(
    data: str,
    *,
    media_type: str,
    stats: ImageShrinkStats,
) -> tuple[str, str] | None:
    base64_bytes = _base64_size(data)
    if base64_bytes <= stats.max_base64_bytes:
        stats.skipped_count += 1
        return None

    stats.candidates += 1
    stats.original_base64_bytes_max = max(stats.original_base64_bytes_max, base64_bytes)
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:  # noqa: BLE001 - malformed payload should not crash recovery
        _record_error(stats, "decode_failed")
        return None

    if _load_pillow_image() is None:
        _record_error(stats, "pillow_unavailable")
        return None

    shrunk = _resize_image_bytes(
        raw,
        media_type=media_type,
        original_base64_bytes=base64_bytes,
        target_base64_bytes=stats.target_base64_bytes,
    )
    if shrunk is None:
        _record_error(stats, "resize_failed")
        return None

    shrunk_base64, shrunk_media_type = shrunk
    shrunk_bytes = _base64_size(shrunk_base64)
    if shrunk_bytes >= base64_bytes or shrunk_bytes > stats.target_base64_bytes:
        _record_error(stats, "target_not_reached")
        return None

    stats.changed_count += 1
    stats.shrunk_base64_bytes_max = max(stats.shrunk_base64_bytes_max, shrunk_bytes)
    return shrunk_base64, shrunk_media_type


def _resize_image_bytes(
    raw: bytes,
    *,
    media_type: str,
    original_base64_bytes: int,
    target_base64_bytes: int,
) -> tuple[str, str] | None:
    pil_image = _load_pillow_image()
    if pil_image is None:
        return None

    try:
        with pil_image.open(BytesIO(raw)) as image:
            image.load()
            source = image.copy()
            original_format = _format_for_media_type(media_type) or image.format or "PNG"
            attempts = _encode_attempts(original_format=original_format)
    except Exception:  # noqa: BLE001 - invalid image or unsupported decoder
        return None

    best: tuple[str, str] | None = None
    best_size = original_base64_bytes
    for fmt, output_media_type, quality in attempts:
        scale = min(0.95, math.sqrt(target_base64_bytes / max(original_base64_bytes, 1)) * 0.95)
        scale = min(0.95, max(0.05, scale))
        for _ in range(10):
            encoded = _encode_resized_image(
                source,
                fmt=fmt,
                scale=scale,
                quality=quality,
            )
            if encoded is None:
                break
            encoded_size = _base64_size(encoded)
            if encoded_size < best_size:
                best = (encoded, output_media_type)
                best_size = encoded_size
            if encoded_size <= target_base64_bytes:
                return encoded, output_media_type
            scale *= 0.75
    return best if best is not None and best_size <= target_base64_bytes else None


def _encode_attempts(*, original_format: str) -> list[tuple[str, str, int | None]]:
    normalized = original_format.upper()
    attempts: list[tuple[str, str, int | None]] = []
    if normalized in {"PNG", "JPEG", "WEBP"}:
        attempts.append((normalized, _media_type_for_format(normalized), 85 if normalized in {"JPEG", "WEBP"} else None))
    if normalized != "JPEG":
        attempts.extend(
            [
                ("JPEG", "image/jpeg", 85),
                ("JPEG", "image/jpeg", 72),
                ("JPEG", "image/jpeg", 60),
            ]
        )
    if normalized == "JPEG":
        attempts.extend(
            [
                ("JPEG", "image/jpeg", 72),
                ("JPEG", "image/jpeg", 60),
            ]
        )
    return attempts


def _encode_resized_image(
    image: Any,
    *,
    fmt: str,
    scale: float,
    quality: int | None,
) -> str | None:
    try:
        width, height = image.size
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resample = getattr(_load_pillow_image(), "Resampling", None)
        resample_filter = getattr(resample, "LANCZOS", 1) if resample is not None else 1
        resized = image.resize(new_size, resample_filter)
        save_kwargs: dict[str, Any] = {"format": fmt, "optimize": True}
        if fmt == "JPEG":
            resized = _to_jpeg_compatible(resized)
            if quality is not None:
                save_kwargs["quality"] = quality
            save_kwargs["progressive"] = True
        elif fmt == "WEBP" and quality is not None:
            save_kwargs["quality"] = quality
        out = BytesIO()
        resized.save(out, **save_kwargs)
        return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001 - try next format/scale
        return None


def _to_jpeg_compatible(image: Any) -> Any:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA", "P"}:
        pil_image = _load_pillow_image()
        rgba = image.convert("RGBA")
        background = pil_image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _load_pillow_image() -> Any | None:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    return Image


def _format_for_media_type(media_type: str) -> str | None:
    mapping = {
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/webp": "WEBP",
        "image/bmp": "BMP",
        "image/gif": "GIF",
    }
    return mapping.get(media_type.lower())


def _media_type_for_format(fmt: str) -> str:
    mapping = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
    }
    return mapping.get(fmt.upper(), "image/jpeg")


def _base64_size(data: str) -> int:
    try:
        return len(data.encode("ascii"))
    except UnicodeEncodeError:
        return len(data)


def _record_error(stats: ImageShrinkStats, reason: str) -> None:
    stats.error_count += 1
    if reason not in stats.error_reasons:
        stats.error_reasons.append(reason)
