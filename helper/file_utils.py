import os
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image

ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
ALLOWED_IMAGE_MIME_TYPES = {'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'}


def check_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def validate_image_file(filepath: str) -> Tuple[bool, Optional[str]]:
    if not filepath or not os.path.exists(filepath):
        return False, "File does not exist"
    
    ext = Path(filepath).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"File extension '{ext}' is not allowed. Allowed extensions: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
    
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, f"File is not a valid image: {str(e)}"
