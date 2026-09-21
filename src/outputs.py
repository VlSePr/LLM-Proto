"""
Run outputs meant for a human: interactive plots, figures, reports.

Notebook-facing helpers that write a file to ``<run>/outputs/``, copy it to the run's Google
Drive folder (with a progress bar for large files) and, on Colab, hand it to the browser as a
download. Everything past the local write is best effort: a Drive or download problem is
reported but never loses the local file.
"""

import os
import sys
import zipfile

from .progress import human_size

OUTPUTS_DIRNAME = "outputs"
GDRIVE_OUTPUTS_SUBDIR = "outputs"


def run_output_dir(checkpoint_dir: str) -> str:
    """``outputs/`` next to *checkpoint_dir* (``checkpoints`` -> ``./outputs``), created if missing."""
    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_dir)), OUTPUTS_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, filename: str, output_dir: str, dpi: int = 150) -> str:
    """Write a matplotlib figure to ``<output_dir>/<filename>`` (PNG by extension) and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def offer_download(path: str) -> bool:
    """Trigger a browser download of *path* on Colab. Returns True if the request was sent.

    A no-op elsewhere: outside Colab the file is already on the machine you are looking at.
    """
    if "google.colab" not in sys.modules:
        return False
    try:
        from google.colab import files
        files.download(path)
        return True
    except Exception as e:
        print(f"  ! Browser download failed ({e}); fetch it from Drive or the Colab file browser.")
        return False


def publish_output(
    local_path: str,
    gdrive_folder_id: str = "",
    gdrive_credentials_path: str = "",
    *,
    download: bool = True,
) -> str:
    """Report, back up and (on Colab) download a finished output file. Returns *local_path*.

    Copies to ``<gdrive_folder_id>/outputs/`` when a Drive folder is given.
    """
    print(f"Saved {local_path} ({human_size(os.path.getsize(local_path))})")

    if gdrive_folder_id:
        try:
            from .gdrive import resolve_subfolder, upload_to_gdrive
            folder = resolve_subfolder(gdrive_folder_id, GDRIVE_OUTPUTS_SUBDIR, gdrive_credentials_path)
            upload_to_gdrive(local_path, folder, gdrive_credentials_path)
            print(f"  -> Copied to Google Drive: {gdrive_folder_id}/{GDRIVE_OUTPUTS_SUBDIR}/")
        except Exception as e:
            print(f"  ! Google Drive copy failed: {e}")

    if download and offer_download(local_path):
        print("  -> Browser download started (allow pop-ups/downloads if nothing happens).")
    return local_path


def list_outputs(output_dir: str) -> list[tuple[str, int]]:
    """``[(filename, bytes), ...]`` of the files in *output_dir*, sorted by name (empty if absent)."""
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        (name, os.path.getsize(os.path.join(output_dir, name)))
        for name in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, name))
    )


def zip_outputs(output_dir: str, zip_path: str | None = None) -> str:
    """Zip every file in *output_dir* (default ``<output_dir>.zip``) and return the zip's path.

    Raises ``FileNotFoundError`` when there is nothing to zip.
    """
    files = list_outputs(output_dir)
    if not files:
        raise FileNotFoundError(f"no output files in {output_dir}")
    zip_path = zip_path or os.path.abspath(output_dir.rstrip("/\\")) + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, _ in files:
            zf.write(os.path.join(output_dir, name), arcname=name)
    return zip_path
