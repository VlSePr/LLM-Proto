"""
Google Drive backup for checkpoints and tokenized data.

Two modes (chosen automatically based on runtime environment):
  - **Colab**: mounts Google Drive via ``drive.mount()`` and copies files
    to ``/content/drive/MyDrive/<folder_id>/``.  No API credentials needed;
    uses the authenticated Colab session directly.
    ``folder_id`` is the **folder name** under My Drive (created automatically).
    Nested names such as ``"LLM/tokenized"`` are allowed.
  - **Local / vast.ai**: uses the Drive REST API v3 with a service-account JSON
    or Application Default Credentials. This requires a one-time credential setup
    but works anywhere (SSH servers, CI, cloud VMs).
    ``folder_id`` is the real Drive folder **ID** (the 33-char hash from the URL).

``resolve_subfolder`` hides the difference: given a parent folder and a name it
returns whatever the current mode uses as a folder handle (a nested path on
Colab, a folder ID in API mode).
"""

import glob
import os
import shutil
import sys
from datetime import datetime, timezone

_COLAB_MOUNT = "/content/drive"


def default_run_folder_name(label: str) -> str:
    """A timestamped Drive subfolder name identifying a training run, e.g. ``'tiny-20260916-143512'``.

    Used to nest a run's checkpoints, config snapshots, metric history, tokenizer and tokenized-data
    cache under one folder instead of a flat shared ``gdrive_folder_id``.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{label}-{ts}"


# ──────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────

def _is_colab() -> bool:
    return "google.colab" in sys.modules


def _ensure_colab_mount() -> None:
    """Mount Drive at ``/content/drive`` if it is not mounted yet."""
    from google.colab import drive
    if not os.path.ismount(_COLAB_MOUNT):
        drive.mount(_COLAB_MOUNT)


def _colab_folder(folder_id: str, create: bool = True) -> str:
    """Mount Drive (once) and return the local path for *folder_id*.

    With ``create=False`` the directory is not created, so callers can probe
    for an existing folder without leaving empty directories on Drive.
    """
    _ensure_colab_mount()
    path = os.path.join(_COLAB_MOUNT, "MyDrive", folder_id)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


# ──────────────────────────────────────────────
# Drive API helpers  (non-Colab only)
# ──────────────────────────────────────────────

# Singleton: cache the Drive v3 service to avoid repeated OAuth handshakes.
# Building the service involves HTTP calls to discover the API schema,
# so reusing it across uploads/downloads saves significant latency.
_drive_service = None


def _get_service(credentials_path: str):
    """Build and cache the Drive v3 API service (singleton pattern)."""
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    if credentials_path and os.path.isfile(credentials_path):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES,
        )
    else:
        import google.auth
        creds, _ = google.auth.default(scopes=SCOPES)

    _drive_service = build("drive", "v3", credentials=creds)
    return _drive_service


def reset_service():
    """Clear the cached Drive service so the next call re-authenticates."""
    global _drive_service
    _drive_service = None


def _find_file(service, name: str, folder_id: str) -> str | None:
    """Return file ID if *name* exists in *folder_id*, else None."""
    query = (
        f"'{folder_id}' in parents and name = '{name}' "
        f"and trashed = false"
    )
    resp = service.files().list(q=query, fields="files(id)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


_FOLDER_MIME = "application/vnd.google-apps.folder"


def _find_folder(service, name: str, parent_id: str) -> str | None:
    """Return the ID of sub-folder *name* under *parent_id*, else None.

    Drive allows several folders with the same name; the oldest one is
    returned so repeated calls are deterministic.
    """
    query = (
        f"'{parent_id}' in parents and name = '{name}' "
        f"and mimeType = '{_FOLDER_MIME}' and trashed = false"
    )
    resp = (
        service.files()
        .list(q=query, fields="files(id)", orderBy="createdTime")
        .execute()
    )
    files = resp.get("files", [])
    return files[0]["id"] if files else None


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def resolve_subfolder(
    parent_folder_id: str,
    name: str,
    credentials_path: str = "",
    create: bool = True,
) -> str | None:
    """
    Return a folder handle for sub-folder *name* inside *parent_folder_id*.

    Colab — the nested folder name ``"<parent>/<name>"`` (usable as ``folder_id``
    in every other function here).  API mode — the Drive folder ID.

    With ``create=True`` the folder is created when missing; with
    ``create=False`` a missing folder yields ``None`` and nothing is created.
    """
    # ── Colab: nested directory under MyDrive ──
    if _is_colab():
        nested = f"{parent_folder_id}/{name}"
        path = _colab_folder(nested, create=create)
        if not create and not os.path.isdir(path):
            return None
        return nested

    # ── API mode ──
    service = _get_service(credentials_path)
    folder_id = _find_folder(service, name, parent_folder_id)
    if folder_id or not create:
        return folder_id
    metadata = {"name": name, "mimeType": _FOLDER_MIME, "parents": [parent_folder_id]}
    result = service.files().create(body=metadata, fields="id").execute()
    return result["id"]


def find_latest_run_folder(
    parent_folder_id: str,
    label: str,
    credentials_path: str = "",
) -> str | None:
    """Most recently created ``"<label>-<timestamp>"`` subfolder under *parent_folder_id*, or ``None``.

    ``default_run_folder_name`` nests a fresh run's checkpoints under a new timestamped subfolder
    of the stable top-level Drive folder. This lets a resume that's still pointed at that stable
    folder (rather than the exact subfolder the original run printed) find the right checkpoint
    instead of silently starting a new model from scratch.
    """
    prefix = f"{label}-"

    if _is_colab():
        parent_path = _colab_folder(parent_folder_id, create=False)
        if not os.path.isdir(parent_path):
            return None
        candidates = sorted(
            name for name in os.listdir(parent_path)
            if name.startswith(prefix) and os.path.isdir(os.path.join(parent_path, name))
        )
        return f"{parent_folder_id}/{candidates[-1]}" if candidates else None

    service = _get_service(credentials_path)
    query = (
        f"'{parent_folder_id}' in parents and name contains '{prefix}' "
        f"and mimeType = '{_FOLDER_MIME}' and trashed = false"
    )
    resp = (
        service.files()
        .list(q=query, fields="files(id, name)", orderBy="createdTime desc")
        .execute()
    )
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def upload_to_gdrive(
    local_path: str,
    folder_id: str,
    credentials_path: str = "",
) -> str:
    """
    Upload a local file to Google Drive.

    Returns:
        Colab — the destination path on the mounted Drive.
        Non-Colab — the Google Drive file ID.
    """
    filename = os.path.basename(local_path)

    # ── Colab: filesystem copy ──
    if _is_colab():
        dest_dir = _colab_folder(folder_id)
        dest = os.path.join(dest_dir, filename)
        shutil.copy2(local_path, dest)
        return dest

    # ── API mode ──
    from googleapiclient.http import MediaFileUpload

    service = _get_service(credentials_path)
    existing_id = _find_file(service, filename, folder_id)
    # resumable=True enables chunked uploads — if the connection drops mid-transfer,
    # the upload can resume from the last successfully sent chunk instead of restarting.
    # This is essential for large checkpoint files (100 MB+) over unreliable connections.
    media = MediaFileUpload(local_path, resumable=True)

    if existing_id:
        result = (
            service.files()
            .update(fileId=existing_id, media_body=media)
            .execute()
        )
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        result = (
            service.files()
            .create(body=metadata, media_body=media, fields="id")
            .execute()
        )
    return result["id"]


def cleanup_remote_checkpoints(
    folder_id: str,
    keep_n: int,
    credentials_path: str = "",
):
    """
    Remove old ``step_*.pt`` files, keeping the last *keep_n*.
    Special files (latest.pt, best.pt) are never removed, so you always
    have a fast-resume checkpoint and the best-loss checkpoint available.
    Sorted by creation time — oldest checkpoints are deleted first.
    """
    # ── Colab: filesystem cleanup ──
    if _is_colab():
        dest_dir = _colab_folder(folder_id)
        step_files = sorted(glob.glob(os.path.join(dest_dir, "step_*.pt")),
                            key=os.path.getmtime)
        for f in step_files[: len(step_files) - keep_n]:
            os.remove(f)
        return

    # ── API mode ──
    service = _get_service(credentials_path)
    query = (
        f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' "
        f"and trashed = false and name contains 'step_'"
    )
    resp = (
        service.files()
        .list(q=query, fields="files(id, name, createdTime)", orderBy="createdTime")
        .execute()
    )
    files = resp.get("files", [])
    if len(files) <= keep_n:
        return
    for f in files[: len(files) - keep_n]:
        service.files().delete(fileId=f["id"]).execute()


def list_remote_checkpoints(
    folder_id: str,
    credentials_path: str = "",
) -> list[dict]:
    """
    List checkpoint ``.pt`` files on Google Drive.

    Returns:
        Sorted list of dicts with keys: name (+ id, createdTime for API mode).
    """
    # ── Colab: filesystem listing ──
    if _is_colab():
        dest_dir = _colab_folder(folder_id)
        pt_files = sorted(glob.glob(os.path.join(dest_dir, "*.pt")),
                          key=os.path.getmtime)
        return [{"name": os.path.basename(f),
                 "createdTime": str(os.path.getmtime(f))}
                for f in pt_files]

    # ── API mode ──
    service = _get_service(credentials_path)
    query = (
        f"'{folder_id}' in parents "
        f"and mimeType != 'application/vnd.google-apps.folder' "
        f"and trashed = false "
        f"and name contains '.pt'"
    )
    resp = (
        service.files()
        .list(q=query, fields="files(id, name, createdTime)", orderBy="createdTime")
        .execute()
    )
    return resp.get("files", [])


def download_from_gdrive(
    filename: str,
    folder_id: str,
    local_dir: str,
    credentials_path: str = "",
) -> str:
    """
    Download a file from Google Drive to *local_dir*.

    The transfer goes to ``<filename>.part`` and is renamed into place only
    when complete, so an interrupted download never leaves a truncated file
    that looks like the real thing.

    Raises:
        FileNotFoundError: If the file doesn't exist on Drive.
    """
    # ── Colab: filesystem copy ──
    if _is_colab():
        src = os.path.join(_colab_folder(folder_id, create=False), filename)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"'{filename}' not found in Google Drive folder '{folder_id}'"
            )
        os.makedirs(local_dir, exist_ok=True)
        dest = os.path.join(local_dir, filename)
        part = dest + ".part"
        try:
            shutil.copy2(src, part)
            os.replace(part, dest)
        except BaseException:
            if os.path.exists(part):
                os.remove(part)
            raise
        return dest

    # ── API mode ──
    from googleapiclient.http import MediaIoBaseDownload

    service = _get_service(credentials_path)
    file_id = _find_file(service, filename, folder_id)
    if not file_id:
        raise FileNotFoundError(
            f"'{filename}' not found in Google Drive folder {folder_id}"
        )

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    part = local_path + ".part"

    request = service.files().get_media(fileId=file_id)
    try:
        with open(part, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        os.replace(part, local_path)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise

    return local_path


# ──────────────────────────────────────────────
# Notebook helper
# ──────────────────────────────────────────────

def describe_drive_setup(
    folder_id: str,
    credentials_path: str = "",
    *,
    data_dir: str = "",
    extra_folders: tuple = (),
) -> dict:
    """Mount (on Colab) and print what the Drive integration will use; returns the summary.

    Reports the checkpoint folder(s) with their ``.pt`` counts, the optional training
    data folder with its ``.txt`` / ``.jsonl`` counts (Colab only), and in API mode the
    credential status. Never raises: problems are recorded under ``"errors"``.
    """
    info: dict = {"mode": "colab" if _is_colab() else "api", "folders": {}, "errors": []}
    folders = [f for f in (folder_id, *extra_folders) if f]

    if _is_colab():
        try:
            _ensure_colab_mount()
            info["mount"] = _COLAB_MOUNT
            print(f"Google Drive mounted at {_COLAB_MOUNT}")
        except Exception as e:
            info["errors"].append(f"mount failed: {e}")
            print(f"! Google Drive mount failed: {e}")
            return info
        for name in folders:
            path = _colab_folder(name)
            n_pt = len([f for f in os.listdir(path) if f.endswith(".pt")])
            info["folders"][name] = {"path": path, "pt_files": n_pt}
            print(f"Checkpoint folder: {path} ({n_pt} .pt files)")
        if data_dir:
            path = _colab_folder(data_dir, create=False)
            if os.path.isdir(path):
                names = os.listdir(path)
                counts = {ext: len([f for f in names if f.endswith(ext)]) for ext in (".txt", ".jsonl")}
                info["data_dir"] = {"path": path, **counts}
                print(f"Training data folder: {path} ({counts['.txt']} .txt, {counts['.jsonl']} .jsonl)")
            else:
                info["errors"].append(f"data_dir not found: {path}")
                print(f"! Training data folder not found: {path}")
        return info

    if credentials_path and os.path.isfile(credentials_path):
        info["credentials"] = credentials_path
        print(f"Google Drive API credentials: {credentials_path}")
    elif credentials_path:
        info["errors"].append(f"credentials file not found: {credentials_path}")
        print(f"! Credentials file not found: {credentials_path}")
    else:
        info["credentials"] = "application-default"
        print("Google Drive API mode (Application Default Credentials)")
    for name in folders:
        try:
            files = list_remote_checkpoints(name, credentials_path)
            info["folders"][name] = {"pt_files": len(files)}
            print(f"Checkpoint folder {name}: {len(files)} .pt files")
        except Exception as e:
            info["errors"].append(f"{name}: {e}")
            print(f"! Could not list Drive folder {name}: {e}")
    return info
