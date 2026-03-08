"""
Google Drive backup for checkpoints.
Supports service-account JSON or interactive OAuth (Colab-friendly).
"""

import os
from typing import Optional

# Lazy imports — these are only needed when backup is enabled
_drive_service = None


def _get_service(credentials_path: str):
    """Build and cache Google Drive API service."""
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    if credentials_path and os.path.isfile(credentials_path):
        creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
    else:
        # Fall back to default application credentials (e.g. Colab, gcloud auth)
        import google.auth
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )

    _drive_service = build("drive", "v3", credentials=creds)
    return _drive_service


def upload_to_gdrive(
    local_path: str,
    folder_id: str,
    credentials_path: str = "",
) -> str:
    """
    Upload a local file to a Google Drive folder.

    Args:
        local_path: Path to the file on disk.
        folder_id: Google Drive folder ID to upload into.
        credentials_path: Path to service-account JSON (or empty for default creds).

    Returns:
        The Google Drive file ID of the uploaded file.
    """
    from googleapiclient.http import MediaFileUpload

    service = _get_service(credentials_path)
    filename = os.path.basename(local_path)

    # Check if a file with the same name already exists in the folder
    existing_id = _find_file(service, filename, folder_id)

    media = MediaFileUpload(local_path, resumable=True)

    if existing_id:
        # Update existing file (avoids duplicates on re-save of latest.pt / best.pt)
        result = (
            service.files()
            .update(fileId=existing_id, media_body=media)
            .execute()
        )
    else:
        metadata = {
            "name": filename,
            "parents": [folder_id],
        }
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
    Remove old step_*.pt files from the Drive folder, keeping the last *keep_n*.
    Special files (latest.pt, best.pt) are never removed.
    """
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

    to_delete = files[: len(files) - keep_n]
    for f in to_delete:
        service.files().delete(fileId=f["id"]).execute()


def _find_file(service, name: str, folder_id: str) -> Optional[str]:
    """Return file ID if *name* exists in *folder_id*, else None."""
    query = (
        f"'{folder_id}' in parents and name = '{name}' "
        f"and trashed = false"
    )
    resp = service.files().list(q=query, fields="files(id)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def list_remote_checkpoints(
    folder_id: str,
    credentials_path: str = "",
) -> list[dict]:
    """
    List checkpoint files in a Google Drive folder.

    Returns:
        Sorted list of dicts with keys: id, name, createdTime.
        Most recent last.
    """
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
    Download a checkpoint file from Google Drive to a local directory.

    Args:
        filename: Name of the file on Drive (e.g. "latest.pt", "step_500.pt").
        folder_id: Google Drive folder ID.
        local_dir: Local directory to save into.
        credentials_path: Path to service-account JSON (or empty for default creds).

    Returns:
        Local path to the downloaded file.

    Raises:
        FileNotFoundError: If the file doesn't exist in the Drive folder.
    """
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

