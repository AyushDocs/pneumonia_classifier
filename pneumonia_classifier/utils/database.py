import os
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    desc,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from pneumonia_classifier.logger import logging

DB_PATH = os.path.join(os.getcwd(), "data", "patient_history.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String)
    prediction = Column(String)
    confidence = Column(String)
    heatmap_path = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    requester_id = Column(String, default="UNKNOWN")
    requester_ip = Column(String, default="UNKNOWN")

class DriftLog(Base):
    __tablename__ = "drift_logs"
    job_id = Column(String, primary_key=True, index=True)
    mean_val = Column(Float)
    std_val = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)

# Pydantic DTOs
class PredictionDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: Optional[int] = None
    patient_id: str
    prediction: str
    confidence: str
    heatmap_path: Optional[str]
    timestamp: datetime
    requester_id: str = "UNKNOWN"
    requester_ip: str = "UNKNOWN"

class DriftLogDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    job_id: str
    mean_val: float
    std_val: float
    timestamp: datetime

def init_db():
    try:
        Base.metadata.create_all(bind=engine)

        # Migrations/Column Checks (handled by create_all normally, but for existing DB):
        with engine.connect() as conn:
            # Check for new columns in case of existing table
            inspector = text("PRAGMA table_info(predictions)")
            columns = [row[1] for row in conn.execute(inspector).fetchall()]

            if "requester_id" not in columns:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN requester_id TEXT DEFAULT 'UNKNOWN'"))
            if "requester_ip" not in columns:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN requester_ip TEXT DEFAULT 'UNKNOWN'"))
            conn.commit()

        logging.info("Database initialized successfully with SQLAlchemy.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_prediction(patient_id: str, prediction: str, confidence: str, heatmap_path: str, requester_id="UNKNOWN", requester_ip="UNKNOWN"):
    db = SessionLocal()
    try:
        new_pred = Prediction(
            patient_id=patient_id,
            prediction=prediction,
            confidence=confidence,
            heatmap_path=heatmap_path,
            requester_id=requester_id,
            requester_ip=requester_ip,
            timestamp=datetime.now()
        )
        db.add(new_pred)
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"Error saving prediction: {e}")
    finally:
        db.close()

def save_drift_log(job_id: str, mean_val: float, std_val: float):
    db = SessionLocal()
    try:
        new_log = DriftLog(
            job_id=job_id,
            mean_val=mean_val,
            std_val=std_val,
            timestamp=datetime.now()
        )
        db.add(new_log)
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"Error saving drift log: {e}")
    finally:
        db.close()

def get_patient_history(patient_id: str) -> List[PredictionDTO]:
    db = SessionLocal()
    try:
        preds = db.query(Prediction).filter(Prediction.patient_id == patient_id).order_by(desc(Prediction.timestamp)).all()
        return [PredictionDTO.model_validate(p) for p in preds]
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return []
    finally:
        db.close()

def get_all_predictions() -> List[PredictionDTO]:
    db = SessionLocal()
    try:
        preds = db.query(Prediction).order_by(Prediction.timestamp.asc()).all()
        return [PredictionDTO.model_validate(p) for p in preds]
    except Exception as e:
        logging.error(f"Error fetching all predictions: {e}")
        return []
    finally:
        db.close()

def get_drift_metrics() -> List[DriftLogDTO]:
    db = SessionLocal()
    try:
        logs = db.query(DriftLog).order_by(DriftLog.timestamp.asc()).all()
        return [DriftLogDTO.model_validate(log) for log in logs]
    except Exception as e:
        logging.error(f"Error fetching drift metrics: {e}")
        return []
    finally:
        db.close()

def purge_old_records(days: int = 30):
    """Deletes records older than specified `days` from database and local tmp directory."""
    db = SessionLocal()
    try:
        from datetime import timedelta

        from pneumonia_classifier.config import config

        cutoff_date = datetime.now() - timedelta(days=days)

        # Get patient IDs for file cleanup
        old_preds = db.query(Prediction).filter(Prediction.timestamp <= cutoff_date).all()
        patient_ids = [p.patient_id for p in old_preds]

        # Delete from DB
        db.query(Prediction).filter(Prediction.timestamp <= cutoff_date).delete()
        db.query(DriftLog).filter(DriftLog.timestamp <= cutoff_date).delete()

        deleted_count = len(patient_ids)
        db.commit()

        if deleted_count > 0 and os.path.exists(config.REPORT_TEMP_DIR):
            for pid in patient_ids:
                f_path = os.path.join(config.REPORT_TEMP_DIR, f"report_{pid}.pdf")
                if os.path.exists(f_path):
                    os.remove(f_path)
            logging.info(f"Purged {deleted_count} records and reports older than {days} days.")
    except Exception as e:
        db.rollback()
        logging.error(f"Failed to purge old records: {e}")
    finally:
        db.close()
