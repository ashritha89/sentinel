import os
from dotenv import load_dotenv

load_dotenv()   # ← THIS WAS MISSING

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    MONGO_URI = os.environ.get('MONGO_URI')
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
