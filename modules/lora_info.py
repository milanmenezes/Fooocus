import os
import json
import struct
import html

import modules.config
from modules.civitai import SIDECAR_SUFFIX

LORA_PAGE_FILENAME = 'lora_trigger_words.html'


def _find_lora_path(filename: str):
    """Return the absolute path of a LoRA filename across all configured dirs."""
    for folder in modules.config.paths_loras:
        candidate = os.path.abspath(os.path.join(folder, filename))
        if os.path.exists(candidate):
            return candidate
    return None


def _read_sidecar_words(lora_path: str):
    """Trigger words saved by the Civitai downloader, if any."""
    sidecar = lora_path + SIDECAR_SUFFIX
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, 'r', encoding='utf-8') as f:
            data = json.load(f)
        words = data.get('trained_words') or []
        return [w for w in words if isinstance(w, str) and w.strip()]
    except Exception:
        return None


def _read_safetensors_metadata(lora_path: str) -> dict:
    """Read the __metadata__ dict from a .safetensors header."""
    try:
        with open(lora_path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len))
        return header.get('__metadata__', {}) or {}
    except Exception:
        return {}


def _top_training_tags(metadata: dict, limit: int = 15):
    """Best-effort trigger hints from a Kohya LoRA's ss_tag_frequency."""
    raw = metadata.get('ss_tag_frequency')
    if not raw:
        return []
    try:
        freq = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    totals = {}
    for tags in freq.values():
        if isinstance(tags, dict):
            for tag, count in tags.items():
                tag = tag.strip()
                if tag:
                    totals[tag] = totals.get(tag, 0) + (count if isinstance(count, int) else 0)
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [tag for tag, _ in ordered[:limit]]


def get_trigger_words(filename: str):
    """Return (words, source) for a LoRA filename.

    source is one of: 'civitai', 'metadata', or '' when nothing was found.
    """
    lora_path = _find_lora_path(filename)
    if lora_path is None:
        return [], ''

    words = _read_sidecar_words(lora_path)
    if words:
        return words, 'civitai'

    words = _top_training_tags(_read_safetensors_metadata(lora_path))
    if words:
        return words, 'metadata'

    return [], ''


def _render_html() -> str:
    rows = []
    for filename in modules.config.lora_filenames:
        words, source = get_trigger_words(filename)
        if source == 'civitai':
            words_html = ', '.join(f'<code>{html.escape(w)}</code>' for w in words)
            source_html = 'Civitai'
        elif source == 'metadata':
            words_html = ('<em>No explicit trigger words. Top training tags:</em><br>'
                          + ', '.join(f'<code>{html.escape(w)}</code>' for w in words))
            source_html = 'safetensors metadata'
        else:
            words_html = '<span class="none">—</span>'
            source_html = '<span class="none">unknown</span>'
        rows.append(
            f'<tr><td class="name">{html.escape(filename)}</td>'
            f'<td>{words_html}</td><td class="src">{source_html}</td></tr>'
        )

    body = '\n'.join(rows) if rows else (
        '<tr><td colspan="3" class="none">No LoRAs installed.</td></tr>'
    )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Installed LoRAs &amp; Trigger Words</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background: #1f1f1f; color: #e6e6e6; }}
  h1 {{ font-size: 20px; }}
  p.hint {{ color: #9a9a9a; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #3a3a3a; vertical-align: top; }}
  th {{ color: #c9c9c9; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
  td.name {{ font-weight: 600; white-space: nowrap; }}
  td.src {{ color: #9a9a9a; white-space: nowrap; }}
  code {{ background: #2e2e2e; padding: 1px 6px; border-radius: 4px; font-size: 13px; }}
  .none {{ color: #6f6f6f; }}
</style>
</head>
<body>
  <h1>Installed LoRAs &amp; Trigger Words</h1>
  <p class="hint">Trigger words come from Civitai when a LoRA was downloaded via the Civitai button;
     otherwise Fooocus shows the most frequent training tags from the file metadata (approximate).</p>
  <table>
    <thead><tr><th>LoRA</th><th>Trigger words</th><th>Source</th></tr></thead>
    <tbody>
{body}
    </tbody>
  </table>
</body>
</html>'''


def build_lora_trigger_page() -> str:
    """Write the LoRA trigger-words HTML page and return its absolute path."""
    os.makedirs(modules.config.path_outputs, exist_ok=True)
    path = os.path.abspath(os.path.join(modules.config.path_outputs, LORA_PAGE_FILENAME))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(_render_html())
    return path
