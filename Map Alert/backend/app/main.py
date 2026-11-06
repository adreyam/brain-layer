from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, String, Float, DateTime, JSON, select, event, text
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session

class Settings(BaseSettings):
    database_url: str = "sqlite:///./citypulse.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    class Config:
        env_file = ".env"

settings = Settings()
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
    poolclass=StaticPool if settings.database_url in {"sqlite://", "sqlite:///:memory:"} else None,
    pool_pre_ping=True,
)
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def sqlite_fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

class Base(DeclarativeBase): pass
class Domain(str, Enum):
    security="security"; mobility="mobility"; energy="energy"; environment="environment"; emergency="emergency"; disaster="disaster"
class Severity(str, Enum):
    low="low"; medium="medium"; high="high"; critical="critical"
class Status(str, Enum):
    active="active"; investigating="investigating"; resolved="resolved"

class Incident(Base):
    __tablename__="incidents"
    id: Mapped[str]=mapped_column(String(36), primary_key=True, default=lambda:str(uuid4()))
    title: Mapped[str]=mapped_column(String(180), index=True)
    description: Mapped[str]=mapped_column(String(2000), default="")
    domain: Mapped[str]=mapped_column(String(32), index=True)
    severity: Mapped[str]=mapped_column(String(16), index=True)
    status: Mapped[str]=mapped_column(String(24), default="active", index=True)
    latitude: Mapped[float]=mapped_column(Float)
    longitude: Mapped[float]=mapped_column(Float)
    source: Mapped[str]=mapped_column(String(120), default="citypulse-demo")
    external_id: Mapped[str|None]=mapped_column(String(180), nullable=True, unique=True)
    occurred_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=lambda:datetime.now(timezone.utc), index=True)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True), default=lambda:datetime.now(timezone.utc))
    metadata_json: Mapped[dict]=mapped_column("metadata", JSON, default=dict)

# PostGIS column is added when running against PostgreSQL; coordinates remain portable for tests and local development.
Base.metadata.create_all(engine)
if engine.dialect.name == "postgresql":
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.execute(text("ALTER TABLE incidents ADD COLUMN IF NOT EXISTS geom geography(Point,4326)"))
        conn.execute(text("UPDATE incidents SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude),4326)::geography WHERE geom IS NULL"))
SessionLocal=sessionmaker(bind=engine, expire_on_commit=False)

class IncidentIn(BaseModel):
    model_config=ConfigDict(extra="forbid")
    title: str=Field(min_length=2,max_length=180)
    description: str=Field(default="",max_length=2000)
    domain: Domain
    severity: Severity
    status: Status=Status.active
    latitude: float=Field(ge=-90,le=90)
    longitude: float=Field(ge=-180,le=180)
    source: str=Field(default="unknown",min_length=1,max_length=120)
    external_id: str|None=Field(default=None,max_length=180)
    occurred_at: datetime|None=None
    metadata: dict[str,Any]=Field(default_factory=dict)

class IncidentOut(BaseModel):
    id:str; title:str; description:str; domain:Domain; severity:Severity; status:Status
    latitude:float; longitude:float; source:str; external_id:str|None; occurred_at:datetime
    created_at:datetime; metadata:dict[str,Any]

def serialize(i: Incident):
    return {"id":i.id,"title":i.title,"description":i.description,"domain":i.domain,"severity":i.severity,"status":i.status,
            "latitude":i.latitude,"longitude":i.longitude,"source":i.source,"external_id":i.external_id,
            "occurred_at":i.occurred_at,"created_at":i.created_at,"metadata":i.metadata_json or {}}
class Hub:
    def __init__(self): self.clients:set[WebSocket]=set()
    async def connect(self, ws):
        await ws.accept(); self.clients.add(ws)
    def disconnect(self, ws): self.clients.discard(ws)
    async def broadcast(self, payload):
        stale=[]
        for ws in self.clients.copy():
            try: await ws.send_json(payload)
            except Exception: stale.append(ws)
        for ws in stale: self.disconnect(ws)
hub=Hub()

app=FastAPI(title="CityPulse Incident API",version="1.0.0",description="Unified incident ingestion and live city operations feed.")
app.add_middleware(CORSMiddleware,allow_origins=[x.strip() for x in settings.cors_origins.split(",")],allow_methods=["*"],allow_headers=["*"])
def db_session():
    db=SessionLocal()
    try: yield db
    finally: db.close()

@app.get("/health")
def health(db:Session=Depends(db_session)):
    db.execute(text("SELECT 1"))
    return {"status":"ok","database":engine.dialect.name}

@app.get("/api/v1/incidents",response_model=list[IncidentOut])
def list_incidents(domain:list[Domain]|None=Query(default=None),severity:list[Severity]|None=Query(default=None),status:list[Status]|None=Query(default=None),limit:int=Query(500,ge=1,le=2000),db:Session=Depends(db_session)):
    stmt=select(Incident).order_by(Incident.occurred_at.desc()).limit(limit)
    if domain: stmt=stmt.where(Incident.domain.in_([x.value for x in domain]))
    if severity: stmt=stmt.where(Incident.severity.in_([x.value for x in severity]))
    if status: stmt=stmt.where(Incident.status.in_([x.value for x in status]))
    return [serialize(x) for x in db.scalars(stmt)]

@app.post("/api/v1/incidents",response_model=IncidentOut,status_code=201)
async def create_incident(data:IncidentIn,db:Session=Depends(db_session)):
    if data.external_id:
        old=db.scalar(select(Incident).where(Incident.external_id==data.external_id))
        if old: raise HTTPException(409,"external_id already exists")
    vals=data.model_dump()
    vals["domain"]=data.domain.value; vals["severity"]=data.severity.value; vals["status"]=data.status.value
    vals["occurred_at"]=data.occurred_at or datetime.now(timezone.utc); vals["metadata_json"]=vals.pop("metadata")
    item=Incident(**vals)
    db.add(item); db.commit(); db.refresh(item)
    if engine.dialect.name=="postgresql":
        db.execute(text("UPDATE incidents SET geom=ST_SetSRID(ST_MakePoint(:lon,:lat),4326)::geography WHERE id=:id"),{"lon":item.longitude,"lat":item.latitude,"id":item.id}); db.commit()
    payload=serialize(item)
    await hub.broadcast({"type":"incident.created","incident":payload})
    return payload

@app.get("/api/v1/incidents/{incident_id}",response_model=IncidentOut)
def get_incident(incident_id:str,db:Session=Depends(db_session)):
    item=db.get(Incident,incident_id)
    if not item: raise HTTPException(404,"incident not found")
    return serialize(item)

@app.patch("/api/v1/incidents/{incident_id}",response_model=IncidentOut)
async def update_incident(incident_id:str,patch:dict[str,Any],db:Session=Depends(db_session)):
    allowed={"status","title","description","severity"}
    if not patch or set(patch)-allowed: raise HTTPException(422,f"Allowed fields: {', '.join(sorted(allowed))}")
    item=db.get(Incident,incident_id)
    if not item: raise HTTPException(404,"incident not found")
    for key,value in patch.items():
        if key in {"status","severity"}:
            enum=Status if key=="status" else Severity
            try: value=enum(value).value
            except ValueError: raise HTTPException(422,f"Invalid {key}")
        setattr(item,key,value)
    db.commit(); db.refresh(item)
    payload=serialize(item); await hub.broadcast({"type":"incident.updated","incident":payload})
    return payload

@app.websocket("/api/v1/ws")
async def websocket_feed(ws:WebSocket):
    await hub.connect(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: hub.disconnect(ws)

@app.get("/api/v1/heatmap")
def heatmap(domain:Domain|None=None,db:Session=Depends(db_session)):
    stmt=select(Incident).where(Incident.status!="resolved")
    if domain: stmt=stmt.where(Incident.domain==domain.value)
    return [{"latitude":i.latitude,"longitude":i.longitude,"weight":{"low":0.35,"medium":0.6,"high":0.8,"critical":1.0}[i.severity],"domain":i.domain} for i in db.scalars(stmt)]

@app.post("/api/v1/demo/seed")
async def seed_demo(db:Session=Depends(db_session)):
    if db.scalar(select(Incident.id).limit(1)): return {"created":0,"message":"Incidents already exist"}
    seeds=[
      ("Traffic slowdown on Avenue Habib Bourguiba","mobility","high",36.8008,10.1800,"Traffic flow agent"),
      ("Intersection signal outage","mobility","medium",36.8065,10.1815,"Traffic flow agent"),
      ("Suspicious activity reported","security","medium",36.8012,10.1650,"Public safety agent"),
      ("Street lighting circuit failure","energy","high",36.7952,10.1710,"Energy monitor"),
      ("Elevated particulate reading","environment","high",36.8120,10.1655,"Air quality agent"),
      ("Medical response requested","emergency","critical",36.7980,10.1880,"Emergency response agent"),
      ("Localized water accumulation","disaster","medium",36.8180,10.1900,"Flood monitor"),
      ("Transformer temperature warning","energy","low",36.8080,10.1530,"Energy monitor"),
      ("Road obstruction after collision","mobility","critical",36.7920,10.1830,"Traffic flow agent"),
      ("Smoke alarm near market","emergency","high",36.8030,10.1735,"Emergency response agent")]
    created=[]
    for title,domain,severity,lat,lon,source in seeds:
        i=Incident(title=title,description="Simulated incident for CityPulse demonstration.",domain=domain,severity=severity,status="active",latitude=lat,longitude=lon,source=source,metadata_json={"simulated":True})
        db.add(i); created.append(i)
    db.commit()
    for i in created: db.refresh(i)
    for i in created:
        if engine.dialect.name=="postgresql": db.execute(text("UPDATE incidents SET geom=ST_SetSRID(ST_MakePoint(:lon,:lat),4326)::geography WHERE id=:id"),{"lon":i.longitude,"lat":i.latitude,"id":i.id})
    db.commit()
    for i in created: await hub.broadcast({"type":"incident.created","incident":serialize(i)})
    return {"created":len(created)}
