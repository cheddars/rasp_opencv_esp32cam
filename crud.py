from datetime import datetime
from sqlalchemy.orm import Session
import models, schemas


def insert_detection_result(db: Session, module_id: str, image_path: str, detection: str, confidence: float):
    now = datetime.now()
    db_image = models.ImageArchive(
                                yyyy=now.strftime('%Y'),
                                mm=now.strftime('%m'),
                                dd=now.strftime('%d'),
                                module=module_id,
                                image_path=image_path,
                                detection=detection,
                                confidence=confidence)

    db.add(db_image)
    db.commit()
    db.refresh(db_image)
