import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import the Flask app instance from app.py
from app import app as application

# Optional alias so both wsgi:application and wsgi:app work
app = application
