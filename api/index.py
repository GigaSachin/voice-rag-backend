import sys
from pathlib import Path

# Add backend folder to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import FastAPI app
from main import app