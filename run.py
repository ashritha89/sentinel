from dotenv import load_dotenv
import os
from app import create_app

# Load environment variables from .env
load_dotenv()

# Get Mongo URI from .env
MONGO_URI_VALUE = os.environ.get("MONGO_URI")
if not MONGO_URI_VALUE:
    raise Exception("❌ MONGO_URI not found in .env file")

config_dict = {
    "SECRET_KEY": os.environ.get("SECRET_KEY", "default_secret"),
    "MONGO_URI": MONGO_URI_VALUE,
    "DEBUG": os.environ.get("FLASK_ENV") == "development"
}

# Create Flask app
app = create_app(config_dict=config_dict)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("✅ Attempting to connect with URI:", MONGO_URI_VALUE)
    print(f"🚀 Starting server on 0.0.0.0:{port}")

    app.run(host="0.0.0.0", port=port, debug=config_dict["DEBUG"])
