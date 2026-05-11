"""
Google Drive backup for checkpoints.

Two modes (chosen automatically based on runtime environment):
  - **Colab**: mounts Google Drive via ``drive.mount()`` and copies files
    to ``/content/drive/MyDrive/<folder_id>/``.  No API credentials needed;
    uses the authenticated Colab session directly.
    ``folder_id`` is the **folder name** under My Drive (created automatically).
  - **Local / vast.ai**: uses the Drive REST API v3 with a service-account JSON
    or Application Default Credentials. This requires a one-time credential setup
    but works anywhere (SSH servers, CI, cloud VMs).
    ``folder_id`` is the real Drive folder **ID** (the 33-char hash from the URL).
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


def upload_dir_to_gdrive(
    local_dir: str,
    folder_id: str,
    file_ext: str = ".bin",
    credentials_path: str = "",
) -> list:
    """
    Upload all files matching *file_ext* from *local_dir* to a Google Drive folder.

    Files are uploaded one by one using the existing ``upload_to_gdrive`` function,
    which supports resumable multi-chunk uploads for large files.

    Returns:
        List of (filename, file_id_or_dest_path) tuples for each uploaded file.
    """
    pattern = os.path.join(local_dir, f"*{file_ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No {file_ext} files found in {local_dir}")
        return []

    results = []
    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"  Uploading {fname} ...", end=" ", flush=True)
        result = upload_to_gdrive(fpath, folder_id, credentials_path=credentials_path)
        print("done")
        results.append((fname, result))

    return results


def download_dir_from_gdrive(
    folder_id: str,
    local_dir: str,
    file_ext: str = ".bin",
    credentials_path: str = "",
    skip_existing: bool = True,
) -> list:
    """
    Download all files matching *file_ext* from a Google Drive folder to *local_dir*.

    Already-present local files are skipped by default (``skip_existing=True``),
    so re-running this function is safe and cheap.

    Returns:
        List of local file paths that were downloaded (skipped files are excluded).
    """
    os.makedirs(local_dir, exist_ok=True)

    # ── Colab: filesystem copy from mounted Drive ──
    if _is_colab():
        src_dir = _colab_folder(folder_id)
        pattern = os.path.join(src_dir, f"*{file_ext}")
        remote_files = sorted(glob.glob(pattern))
        downloaded = []
        for src in remote_files:
            fname = os.path.basename(src)
            dest = os.path.join(local_dir, fname)
            if skip_existing and os.path.exists(dest):
                print(f"  Skipping {fname} (already exists)")
                continue
            print(f"  Copying {fname} ...", end=" ", flush=True)
            shutil.copy2(src, dest)
            print("done")
            downloaded.append(dest)
        return downloaded

    # ── API mode ──
    import io
    from googleapiclient.http import MediaIoBaseDownload

    service = _get_service(credentials_path)

    # List all files in the folder matching the extension
    # Drive has no glob, so we fetch all non-folder files and filter by name suffix.
    query = (
        f"'{folder_id}' in parents "
        f"and mimeType != 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    resp = service.files().list(
        q=query, fields="files(id, name)", orderBy="name"
    ).execute()
    remote_files = [f for f in resp.get("files", []) if f["name"].endswith(file_ext)]

    if not remote_files:
        print(f"  No {file_ext} files found in Drive folder {folder_id}")
        return []

    downloaded = []
    for f in remote_files:
        fname = f["name"]
        dest = os.path.join(local_dir, fname)
        if skip_existing and os.path.exists(dest):
            print(f"  Skipping {fname} (already exists)")
            continue
        print(f"  Downloading {fname} ...", end=" ", flush=True)
        request = service.files().get_media(fileId=f["id"])
        with open(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        print("done")
        downloaded.append(dest)

    return downloaded

