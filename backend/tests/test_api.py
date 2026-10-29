from app import settings


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert "zerogpt_available" in body


def test_detect_ai_text(client):
    response = client.post("/api/detect", json={"text": "SYNTHETIC_MARKER hello world."})
    assert response.status_code == 200
    body = response.json()
    assert body["ai_percent"] == 92.0
    assert body["verdict"] == "ai"
    assert body["zerogpt"]["pct"] == 88.0
    assert body["source"] == "text"
    assert isinstance(body["id"], int)
    assert body["sentences"][0]["score"] == 92.0


def test_detect_human_text(client):
    response = client.post("/api/detect", json={"text": "A normal human sentence here."})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "human"
    assert body["ai_percent"] == 8.0


def test_detect_empty_text(client):
    response = client.post("/api/detect", json={"text": ""})
    assert response.status_code == 422


def test_detect_whitespace_text(client):
    response = client.post("/api/detect", json={"text": "   \n  "})
    assert response.status_code == 400


def test_detect_too_long_rejected(client):
    response = client.post("/api/detect", json={"text": "x" * (settings.MAX_REQUEST_CHARS + 1)})
    assert response.status_code == 422


def test_detect_long_input_truncated_with_warning(client):
    text = "SYNTHETIC_MARKER word here. " * 2000
    response = client.post("/api/detect", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert len(body["text"]) <= settings.MAX_ANALYSIS_CHARS
    assert any("limited to the first" in w for w in body["warnings"])


def test_file_txt(client):
    files = {"file": ("notes.txt", "plain human notes", "text/plain")}
    response = client.post("/api/detect/file", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "file"
    assert body["filename"] == "notes.txt"
    assert body["verdict"] == "human"


def test_file_txt_ai_marker(client):
    files = {"file": ("essay.txt", "SYNTHETIC_MARKER generated essay", "text/plain")}
    response = client.post("/api/detect/file", files=files)
    assert response.status_code == 200
    assert response.json()["verdict"] == "ai"


def test_file_unsupported_type(client):
    files = {"file": ("doc.rtf", b"binary", "application/rtf")}
    response = client.post("/api/detect/file", files=files)
    assert response.status_code == 415


def test_file_empty_document(client):
    files = {"file": ("empty.txt", "   \n\n ", "text/plain")}
    response = client.post("/api/detect/file", files=files)
    assert response.status_code == 422


def test_file_too_large(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 8)
    files = {"file": ("big.txt", "0123456789abcdef", "text/plain")}
    response = client.post("/api/detect/file", files=files)
    assert response.status_code == 413


def test_history_flow(client):
    first = client.post("/api/detect", json={"text": "SYNTHETIC_MARKER one"}).json()
    second = client.post("/api/detect", json={"text": "human one"}).json()

    listing = client.get("/api/history").json()
    assert listing["total"] == 2
    assert {item["id"] for item in listing["items"]} == {first["id"], second["id"]}
    assert all("result_json" not in item for item in listing["items"])

    detail = client.get(f"/api/history/{first['id']}")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["ai_percent"] == 92.0
    assert detail_body["sentences"]

    deleted = client.delete(f"/api/history/{first['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/api/history/{first['id']}").status_code == 404

    clear = client.delete("/api/history")
    assert clear.json()["deleted"] == 1
    assert client.get("/api/history").json()["total"] == 0


def test_history_missing_item(client):
    assert client.get("/api/history/99999").status_code == 404
    assert client.delete("/api/history/99999").status_code == 404
