import os
import re

import httpx
from tqdm import tqdm

import modules.config

# Read the civitai.red / civitai.com API key from the environment.
# Set CIVITAI_API_KEY in your Colab Enterprise environment before launching.
CIVITAI_API_KEY = os.environ.get('CIVITAI_API_KEY', '')

# API + download host. civitai.red is a drop-in mirror of civitai.com.
CIVITAI_DOMAIN = 'https://civitai.red'


def _auth_headers():
    # Prefer a live read of the env var (Colab may set it after import),
    # falling back to the value captured at import time.
    api_key = os.environ.get('CIVITAI_API_KEY', '') or CIVITAI_API_KEY
    if api_key:
        return {'Authorization': f'Bearer {api_key}'}
    return {}


def parse_civitai_url(url: str):
    """Extract (model_id, version_id) from a Civitai model URL.

    Supports:
      - https://civitai.red/models/12345
      - https://civitai.red/models/12345/some-name
      - https://civitai.red/models/12345?modelVersionId=67890
      - https://civitai.com/... (same shapes)
    Either value may be None if not present in the URL.
    """
    url = url.strip()
    model_match = re.search(r'/models/(\d+)', url)
    version_match = re.search(r'[?&]modelVersionId=(\d+)', url)
    model_id = model_match.group(1) if model_match else None
    version_id = version_match.group(1) if version_match else None
    return model_id, version_id


def _pick_model_file(files: list) -> dict:
    """Choose the primary model checkpoint file from a version's file list."""
    model_files = [f for f in files if f.get('type') == 'Model'] or files
    if not model_files:
        raise ValueError('No downloadable model file found for this version.')
    for f in model_files:
        if f.get('primary'):
            return f
    return model_files[0]


def resolve_lora(url: str):
    """Resolve a Civitai model URL to (download_url, file_name)."""
    model_id, version_id = parse_civitai_url(url)
    if model_id is None and version_id is None:
        raise ValueError(f'Could not find a Civitai model id in URL: {url!r}')

    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        if version_id is None:
            resp = client.get(
                f'{CIVITAI_DOMAIN}/api/v1/models/{model_id}',
                headers=_auth_headers(),
            )
            resp.raise_for_status()
            versions = resp.json().get('modelVersions') or []
            if not versions:
                raise ValueError('This model has no versions to download.')
            version = versions[0]
        else:
            resp = client.get(
                f'{CIVITAI_DOMAIN}/api/v1/model-versions/{version_id}',
                headers=_auth_headers(),
            )
            resp.raise_for_status()
            version = resp.json()

    model_file = _pick_model_file(version.get('files') or [])
    file_name = model_file.get('name')
    download_url = model_file.get('downloadUrl') or version.get('downloadUrl')
    if not download_url:
        raise ValueError('Civitai API did not return a download URL.')
    if not file_name:
        file_name = f'civitai_{version.get("id", "lora")}.safetensors'
    return download_url, file_name


def download_lora(url: str) -> str:
    """Download a LoRA from a Civitai model URL into the loras folder.

    Returns the saved file name. Skips the download if the file already exists.
    """
    download_url, file_name = resolve_lora(url)
    model_dir = modules.config.paths_loras[0]
    os.makedirs(model_dir, exist_ok=True)
    dest = os.path.abspath(os.path.join(model_dir, file_name))

    if os.path.exists(dest):
        print(f'LoRA already present, skipping download: {dest}')
        return file_name

    print(f'Downloading Civitai LoRA: "{download_url}" -> {dest}')
    tmp = dest + '.part'
    with httpx.stream('GET', download_url, headers=_auth_headers(),
                      follow_redirects=True, timeout=None) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get('Content-Length', 0)) or None
        with open(tmp, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True,
                                        desc=file_name) as bar:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)
    return file_name
