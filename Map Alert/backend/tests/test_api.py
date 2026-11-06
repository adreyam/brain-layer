import os
os.environ["DATABASE_URL"]="sqlite://"
from fastapi.testclient import TestClient
from app.main import app, Base, engine, SessionLocal
import pytest

@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine); Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

def payload(**overrides):
    data={"title":"Road blockage","description":"One lane blocked","domain":"mobility","severity":"high","latitude":36.8,"longitude":10.18,"source":"test-agent","external_id":"test-001"}
    data.update(overrides); return data

def test_ingest_get_and_filter():
    with TestClient(app) as client:
        response=client.post("/api/v1/incidents",json=payload())
        assert response.status_code==201
        assert response.json()["domain"]=="mobility"
        assert client.get("/api/v1/incidents",params={"domain":"mobility","severity":"high"}).json()[0]["title"]=="Road blockage"
        assert client.get("/api/v1/incidents/"+response.json()["id"]).status_code==200

def test_duplicate_external_id_conflicts():
    with TestClient(app) as client:
        assert client.post("/api/v1/incidents",json=payload()).status_code==201
        assert client.post("/api/v1/incidents",json=payload()).status_code==409

def test_validation_and_update():
    with TestClient(app) as client:
        assert client.post("/api/v1/incidents",json=payload(latitude=200)).status_code==422
        row=client.post("/api/v1/incidents",json=payload(external_id="unique")).json()
        assert client.patch("/api/v1/incidents/"+row["id"],json={"status":"resolved"}).json()["status"]=="resolved"
        assert client.patch("/api/v1/incidents/"+row["id"],json={"latitude":0}).status_code==422

def test_seed_and_heatmap():
    with TestClient(app) as client:
        assert client.post("/api/v1/demo/seed").json()["created"]==10
        assert len(client.get("/api/v1/heatmap",params={"domain":"energy"}).json())==2

def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"]=="ok"
