"""API-mode folder resolution in src/gdrive.py, using a fake Drive v3 service."""
import pytest

from src import gdrive


class _Call:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeFiles:
    """Minimal stand-in for ``service.files()`` covering list() and create()."""

    def __init__(self, folders):
        self.folders = folders          # dicts: {id, name, parent, created}
        self.queries = []
        self.created = []

    def list(self, q, fields, orderBy=None):
        self.queries.append({"q": q, "fields": fields, "orderBy": orderBy})
        hits = [f for f in self.folders
                if f"'{f['parent']}' in parents" in q and f"name = '{f['name']}'" in q]
        if orderBy == "createdTime":
            hits.sort(key=lambda f: f["created"])
        return _Call({"files": [{"id": f["id"]} for f in hits]})

    def create(self, body, fields, **kw):
        new = {"id": f"new-{len(self.created)}", "name": body["name"],
               "parent": body["parents"][0], "created": 99, "mime": body["mimeType"]}
        self.created.append(new)
        self.folders.append(new)
        return _Call({"id": new["id"]})


class FakeService:
    def __init__(self, folders=None):
        self._files = FakeFiles(folders or [])

    def files(self):
        return self._files


@pytest.fixture
def api_mode(monkeypatch):
    monkeypatch.setattr(gdrive, "_is_colab", lambda: False)
    yield
    gdrive.reset_service()


def _install(monkeypatch, service):
    monkeypatch.setattr(gdrive, "_get_service", lambda creds: service)
    return service


def test_resolve_subfolder_finds_existing_folder(api_mode, monkeypatch):
    svc = _install(monkeypatch, FakeService([
        {"id": "younger", "name": "tokenized", "parent": "root", "created": 2},
        {"id": "older", "name": "tokenized", "parent": "root", "created": 1},
    ]))
    assert gdrive.resolve_subfolder("root", "tokenized") == "older"   # oldest wins, deterministic
    assert svc.files().created == []


def test_resolve_subfolder_creates_when_missing(api_mode, monkeypatch):
    svc = _install(monkeypatch, FakeService())
    assert gdrive.resolve_subfolder("root", "tokenized") == "new-0"
    assert svc.files().created[0]["mime"] == "application/vnd.google-apps.folder"
    assert svc.files().created[0]["parent"] == "root"
    assert gdrive.resolve_subfolder("root", "tokenized") == "new-0"    # found now, not re-created
    assert len(svc.files().created) == 1


def test_resolve_subfolder_returns_none_when_missing_and_create_false(api_mode, monkeypatch):
    svc = _install(monkeypatch, FakeService())
    assert gdrive.resolve_subfolder("root", "tokenized", create=False) is None
    assert svc.files().created == []


def test_find_folder_query_filters_folder_mimetype(api_mode, monkeypatch):
    svc = _install(monkeypatch, FakeService())
    gdrive.resolve_subfolder("root", "abc", create=False)
    q = svc.files().queries[0]
    assert "mimeType = 'application/vnd.google-apps.folder'" in q["q"]
    assert "trashed = false" in q["q"]
    assert q["orderBy"] == "createdTime"


def test_describe_drive_setup_api_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(gdrive, "list_remote_checkpoints", lambda folder, creds="": [{"name": "latest.pt"}])
    info = gdrive.describe_drive_setup("FOLDER", "", extra_folders=("FOLDER/expert",))
    assert info["mode"] == "api" and info["credentials"] == "application-default"
    assert info["folders"] == {"FOLDER": {"pt_files": 1}, "FOLDER/expert": {"pt_files": 1}}
    assert info["errors"] == []

    def boom(folder, creds=""):
        raise RuntimeError("no network")
    monkeypatch.setattr(gdrive, "list_remote_checkpoints", boom)
    info = gdrive.describe_drive_setup("FOLDER", str(tmp_path / "missing.json"))
    assert info["folders"] == {} and len(info["errors"]) == 2
