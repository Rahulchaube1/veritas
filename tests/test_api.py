import pytest
from fastapi.testclient import TestClient

from veritas import Veritas
from veritas.api import create_app
from veritas.integrations import searxng


def test_service():
    with TestClient(create_app(Veritas())) as client:
        data = {"documents": [{"id": "entry", "title": "Water", "text": "The lake is cold."}]}
        assert client.post("/v1/index/documents", json=data).json()["total"] == 1
        assert client.get("/v1/health").json()["status"] == "ok"
        assert client.get("/v1/models").json()["calibration"] is None
        req = {"query": "The lake is cold."}
        for path in ["/v1/search", "/v1/verify", "/v1/evidence/analyze"]:
            res = client.post(path, json=req)
            assert res.status_code == 200
            result = res.json()
            assert result["claims"][0]["verdict"] == "supported"
        key = result["claims"][0]["id"]
        assert client.get(f"/v1/evidence/{key}").json()["query"] == req["query"]
        assert client.post("/v1/retrieve", json=req).json()[0]["doc_id"] == "entry"
        assert client.get("/v1/metrics").json()["queries"] == 3
        assert client.get("/").status_code == 200
        assert client.get("/assets/benchmark.json").json()["rows"]
        assert client.post("/v1/search", json={"query": ""}).status_code == 422
        assert client.post("/v1/search", json={"query": "Tolls as of 2022-02-31"}).status_code == 422
        assert client.post("/v1/index/documents", json=data).status_code == 200
        assert client.get(f"/v1/evidence/{key}").status_code == 404


def test_access_controls():
    with pytest.raises(ValueError):
        create_app(Veritas(), production=True)
    token = "a" * 25
    with TestClient(create_app(Veritas(), token, True)) as client:
        assert client.get("/v1/health").status_code == 401
        headers = {"Authorization": "Bearer " + token}
        assert client.get("/v1/health", headers=headers).status_code == 200
        assert client.post("/v1/search", content="x" * 1000001, headers=headers).status_code == 413
        for i in range(121):
            res = client.get("/v1/health", headers=headers)
        assert res.status_code == 429


@pytest.mark.parametrize(
    "url", ["file:///dev/null", "http://example.test", "https://abc@site.test", "https://site.test?q=token"]
)
def test_remote_import_validation(url):
    with pytest.raises(ValueError):
        searxng(url, "local", 1)
