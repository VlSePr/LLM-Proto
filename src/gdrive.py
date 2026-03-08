"""
Google Drive backup for checkpoints.

Two modes:
  - **Colab**: mounts Google Drive via ``drive.mount()`` and copies files
    to ``/content/drive/MyDrive/<folder_id>/``.  No API credentials needed.
    ``folder_id`` is the **folder name** under My Drive (created automatically).
  - **Local / vast.ai**: uses the Drive REST API with a service-account JSON
    or Application Default Credentials.  ``folder_id`` is the real Drive
    folder **ID** (the hash from the URL).
"""

import os
import sys
import glob
import shutil
from typing import Optional

_COLAB_MOUNT = "/content/drive"


# ──────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────

def _is_colab() -> bool:
    return "google.colab" in sys.modules


def _colab_folder(folder_id: str) -> str:
    """Mount Drive (once) and return the local path for *folder_id*."""
    from google.colab import drive
    if not os.path.ismount(_COLAB_MOUNT):
        drive.mount(_COLAB_MOUNT)
    path = os.path.join(_COLAB_MOUNT, "MyDrive", folder_id)
    os.makedirs(path, exist_ok=True)
    return path


# ──────────────────────────────────────────────
# Drive API helpers  (non-Colab only)
# ──────────────────────────────────────────────

_drive_service = None


def _get_service(credentials_path: str):
    """Build and cache the Drive v3 API service."""
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


def _find_file(service, name: str, folder_id: str) -> Optional[str]:
    """Return file ID if *name* exists in *folder_id*, else None."""
    query = (
        f"'{folder_id}' in parents and name = '{name}' "
        f"and trashed = false"
    )
    resp = service.files().list(q=query, fields="files(id)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

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
    Special files (latest.pt, best.pt) are never removed.
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
    Download a checkpoint file from Google Drive to *local_dir*.

    Raises:
        FileNotFoundError: If the file doesn't exist on Drive.
    """
    # ── Colab: filesystem copy ──
    if _is_colab():
        src = os.path.join(_colab_folder(folder_id), filename)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"'{filename}' not found in Google Drive folder '{folder_id}'"
            )
        os.makedirs(local_dir, exist_ok=True)
        dest = os.path.join(local_dir, filename)
        shutil.copy2(src, dest)
        return dest

    # ── API mode ──
    import io
    from googleapiclient.http import MediaIoBaseDownload

    service = _get_service(credentials_path)
    file_id = _find_file(service, filename, folder_id)
    if not file_id:
        raise FileNotFoundError(
            f"'{filename}' not found in Google Drive folder {folder_id}"
        )

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    request = service.files().get_media(fileId=file_id)
    with open(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return local_path

