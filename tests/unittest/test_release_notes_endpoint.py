from unittest.mock import patch, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pr_agent.servers.release_notes_endpoint import router


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("RELEASE_NOTES_SECRET", "topsecret")
    # patch get_settings().release_notes.output_dir
    from pr_agent.config_loader import get_settings
    get_settings().release_notes.output_dir = str(tmp_path)
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_rejects_missing_auth(client):
    r = client.post("/generate-release-notes?tag=2026.06.7&previous_tag=2026.06.6&project_id=42")
    assert r.status_code == 401


def test_rejects_wrong_auth(client):
    r = client.post(
        "/generate-release-notes?tag=2026.06.7&previous_tag=2026.06.6&project_id=42",
        headers={"X-Auth": "wrong"},
    )
    assert r.status_code == 401


def test_rejects_missing_params(client):
    r = client.post(
        "/generate-release-notes?tag=2026.06.7",  # missing previous_tag and project_id
        headers={"X-Auth": "topsecret"},
    )
    assert r.status_code == 400


def test_accepts_valid_request_and_enqueues(client, tmp_path):
    with patch(
        "pr_agent.servers.release_notes_endpoint.run_release_notes_tag",
        new=AsyncMock(),
    ) as mock_run:
        r = client.post(
            "/generate-release-notes?tag=2026.06.7&previous_tag=2026.06.6&project_id=42",
            headers={"X-Auth": "topsecret"},
        )
    assert r.status_code == 202
    assert r.json() == {"status": "accepted", "tag": "2026.06.7"}


def test_skips_when_marker_exists(client, tmp_path):
    (tmp_path / ".published-2026.06.7").write_text("")
    r = client.post(
        "/generate-release-notes?tag=2026.06.7&previous_tag=2026.06.6&project_id=42",
        headers={"X-Auth": "topsecret"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "already_published"
