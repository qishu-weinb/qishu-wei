from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'data.db'}"
UPLOAD_DIR = BASE_DIR / "uploads"
API_PREFIX = "/api"
SECRET_KEY = "change-this-secret-before-production"
TOKEN_EXPIRE_SECONDS = 60 * 60 * 24 * 7
MAX_UPLOAD_SIZE = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}
MODEL_PATH = BASE_DIR / "models" / "model_busbusi_multitask_fold1.pt"
MODEL_VERSION = "efficientnet-b0-multitask-v1"
MODEL_INPUT_SIZE = 256
