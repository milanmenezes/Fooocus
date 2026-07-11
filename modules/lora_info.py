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


def _read_sidecar(lora_path: str):
    """Metadata saved by the Civitai downloader, if any."""
    sidecar = lora_path + SIDECAR_SUFFIX
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, 'r', encoding='utf-8') as f:
            data = json.load(f)
        words = data.get('trained_words') or []
        return {
            'words': [w for w in words if isinstance(w, str) and w.strip()],
            'base_model': data.get('base_model') or '',
            'url': data.get('civitai_url') or data.get('source_url') or '',
        }
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


def get_lora_info(filename: str) -> dict:
    """Return {words, source, base_model, url} for a LoRA filename.

    source is one of: 'civitai', 'metadata', or '' when nothing was found.
    """
    info = {'words': [], 'source': '', 'base_model': '', 'url': ''}
    lora_path = _find_lora_path(filename)
    if lora_path is None:
        return info

    sidecar = _read_sidecar(lora_path)
    if sidecar is not None:
        info['base_model'] = sidecar['base_model']
        info['url'] = sidecar['url']
        if sidecar['words']:
            info['words'] = sidecar['words']
            info['source'] = 'civitai'

    if not info['words'] or not info['base_model']:
        metadata = _read_safetensors_metadata(lora_path)
        if not info['base_model']:
            info['base_model'] = metadata.get('ss_base_model_version', '') or ''
        if not info['words']:
            words = _top_training_tags(metadata)
            if words:
                info['words'] = words
                info['source'] = 'metadata'

    return info


def delete_lora(filename: str) -> str:
    """Delete a LoRA file (and its Civitai sidecar). Returns a status message."""
    lora_path = _find_lora_path(filename)
    if lora_path is None:
        return f'Not found: {filename}'

    # Refuse anything that resolves outside the configured LoRA folders.
    allowed = False
    for folder in modules.config.paths_loras:
        folder = os.path.abspath(folder)
        if os.path.commonpath([lora_path, folder]) == folder:
            allowed = True
            break
    if not allowed:
        return f'Refused to delete path outside the LoRA folders: {filename}'

    try:
        os.remove(lora_path)
        sidecar = lora_path + SIDECAR_SUFFIX
        if os.path.exists(sidecar):
            os.remove(sidecar)
    except Exception as e:
        return f'Could not delete {filename}: {e}'
    print(f'Deleted LoRA: {lora_path}')
    return f'Deleted {filename}'


def trigger_words_html(filename) -> str:
    """Small HTML snippet listing a LoRA's trigger words, for the LoRA dropdown rows."""
    if not filename or filename == 'None':
        return ''
    info = get_lora_info(filename)
    if not info['words']:
        return ''
    if info['source'] == 'civitai':
        label = 'Trigger words'
        words = info['words']
    else:
        label = 'Top training tags'
        words = info['words'][:10]
    words_html = ', '.join(f'<code>{html.escape(w)}</code>' for w in words)
    return (f'<div style="font-size: 12px; opacity: .75; margin: -4px 0 4px 2px;">'
            f'{label}: {words_html}</div>')


# The page is served from gradio's "<root>/file=<abspath>" route, so stripping
# everything from "file=" onwards yields the app root for API calls.
_PAGE_SCRIPT = '''
async function deleteLora(btn) {
  const filename = btn.dataset.filename;
  if (!confirm('Delete "' + filename + '" from disk? This cannot be undone.')) {
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Deleting…';
  const root = window.location.pathname.split('file=')[0];
  try {
    const resp = await fetch(root + 'run/delete_lora', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({data: [filename]}),
    });
    if (!resp.ok) {
      throw new Error('HTTP ' + resp.status);
    }
    const result = await resp.json();
    const message = (result.data && result.data[0]) || '';
    if (message.startsWith('Deleted')) {
      btn.closest('tr').remove();
    } else {
      alert(message || 'Delete failed.');
      btn.disabled = false;
      btn.textContent = 'Delete';
    }
  } catch (e) {
    alert('Delete failed: ' + e);
    btn.disabled = false;
    btn.textContent = 'Delete';
  }
}
'''


def _render_html() -> str:
    rows = []
    for filename in modules.config.lora_filenames:
        info = get_lora_info(filename)
        words, source = info['words'], info['source']
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

        if info['base_model']:
            base_html = html.escape(info['base_model'])
        else:
            base_html = '<span class="none">—</span>'

        if info['url']:
            link_html = (f'<a href="{html.escape(info["url"])}" target="_blank" '
                         f'rel="noopener">{html.escape(info["url"])}</a>')
        else:
            link_html = '<span class="none">—</span>'

        rows.append(
            f'<tr><td class="name">{html.escape(filename)}</td>'
            f'<td class="base">{base_html}</td>'
            f'<td>{words_html}</td>'
            f'<td class="link">{link_html}</td>'
            f'<td class="src">{source_html}</td>'
            f'<td><button class="delete" data-filename="{html.escape(filename)}" '
            f'onclick="deleteLora(this)">Delete</button></td></tr>'
        )

    body = '\n'.join(rows) if rows else (
        '<tr><td colspan="6" class="none">No LoRAs installed.</td></tr>'
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
  td.base {{ white-space: nowrap; }}
  td.link {{ max-width: 260px; overflow-wrap: anywhere; font-size: 13px; }}
  td.link a {{ color: #6fb3ff; }}
  code {{ background: #2e2e2e; padding: 1px 6px; border-radius: 4px; font-size: 13px; }}
  .none {{ color: #6f6f6f; }}
  button.delete {{ background: #5a2323; color: #ffb3b3; border: 1px solid #7a3030; border-radius: 4px;
                   padding: 4px 10px; cursor: pointer; font-size: 13px; }}
  button.delete:hover {{ background: #7a2b2b; }}
  button.delete:disabled {{ opacity: .5; cursor: wait; }}
</style>
</head>
<body>
  <h1>Installed LoRAs &amp; Trigger Words</h1>
  <p class="hint">Trigger words, base model and link come from Civitai when a LoRA was downloaded via the
     Civitai button; otherwise Fooocus falls back to the file's safetensors metadata (approximate).
     Deleting removes the file from disk — use the Refresh Files button in Fooocus afterwards to update
     the LoRA dropdowns.</p>
  <table>
    <thead><tr><th>LoRA</th><th>Base model</th><th>Trigger words</th><th>Link</th><th>Source</th><th></th></tr></thead>
    <tbody>
{body}
    </tbody>
  </table>
<script>
{_PAGE_SCRIPT}
</script>
</body>
</html>'''


def build_lora_trigger_page() -> str:
    """Write the LoRA trigger-words HTML page and return its absolute path."""
    os.makedirs(modules.config.path_outputs, exist_ok=True)
    path = os.path.abspath(os.path.join(modules.config.path_outputs, LORA_PAGE_FILENAME))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(_render_html())
    return path
