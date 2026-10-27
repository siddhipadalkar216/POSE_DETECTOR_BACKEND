import os
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, relationship


Base = declarative_base()
_engine = None
SessionLocal = None

db_session = scoped_session(sessionmaker())


def init_engine_and_session(database_url: Optional[str] = None) -> None:
    global _engine, SessionLocal
    if not database_url:
        database_url = os.getenv("DATABASE_URL", "sqlite:///dev.db")
    echo_sql = os.getenv("SQL_ECHO", "0") == "1"
    print("[DB] Attempting database connection")
    _engine = create_engine(database_url, echo=echo_sql, future=True)
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DB] Connected successfully")
    except Exception as exc:
        print("[DB] Connection failed")
        raise
    SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    db_session.configure(bind=_engine)
    Base.metadata.create_all(_engine)
    print("[DB] Migrations ensured (tables created if missing)")
    # Lightweight compatibility migration for User fields added after the first schema.
    try:
        columns = {column["name"] for column in inspect(_engine).get_columns("users")}
        additions = {
            "email": "VARCHAR(255)",
            "name": "VARCHAR(255)",
            "height_cm": "DOUBLE PRECISION",
            "weight_kg": "DOUBLE PRECISION",
        }
        with _engine.begin() as conn:
            for column_name, sql_type in additions.items():
                if column_name not in columns:
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {sql_type}"))
        print("[DB] User table columns ensured (email, name, height_cm, weight_kg)")
    except Exception:
        print("[DB] Compatibility migration skipped; inspect the database schema.")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    name = Column(String(255), nullable=True)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)


class Exercise(Base):
    __tablename__ = "exercises"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    metadata_json = Column(JSON, nullable=True)
    default_tolerance = Column(Float, default=15.0)

    attempts = relationship("AttemptLog", back_populates="exercise")


class AttemptLog(Base):
    __tablename__ = "attempt_logs"

    id = Column(Integer, primary_key=True)
    exercise_id = Column(Integer, ForeignKey("exercises.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    score = Column(Float, default=0.0, nullable=False)
    suggestions_json = Column(JSON, nullable=True)
    landmarks_json = Column(JSON, nullable=True)

    exercise = relationship("Exercise", back_populates="attempts")
    user = relationship("User")
