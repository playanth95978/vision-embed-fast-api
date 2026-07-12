import uuid
from sqlmodel import Session
import numpy as np
from app.models import ImageUpload

def create_random_image(db: Session, description: str = "Test image") -> ImageUpload:
    image = ImageUpload(
        description=description,
        image_url=f"/uploads/{uuid.uuid4()}_test.jpg",
        embedding=np.random.rand(512).tolist()
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image
