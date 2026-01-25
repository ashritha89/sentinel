from flask import Flask, request, jsonify
from config import Config
from models import mongo
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
import pickle
import cv2
from skimage.feature import hog
import numpy as np

# ============================
#  LOAD MODEL
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "street_issue_model.pkl")

try:
    MODEL = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ ML Model loaded successfully")
except Exception as e:
    MODEL = None
    print("❌ Model load failed:", e)


# ============================
#  CONSTANTS & STATUS LOGIC
# ============================

BASE_UPLOAD_FOLDER = "static/uploads"

# Final authority mapping (Hyderabad-style)
AUTHORITY_MAP = {
    "Pothole": "GHMC",          
    "Garbage": "GHMC",          
    "Streetlight": "TSSPDCL",   
    "Waterlogging": "HMWSSB",   
    "Unsafe Area": "HYDRA",     
    "Other Urban Issue": "GHMC" 
}

def get_real_world_status(created_at):
    """
    Calculates the status based on a real-world simulation timer.
    - 0 to 10 mins: Report Received
    - 10 to 20 mins: Work Order Dispatched
    - 20+ mins: Issue Resolved
    """
    now = datetime.utcnow()
    # Calculate difference in minutes
    minutes_passed = (now - created_at).total_seconds() / 60 
    
    if minutes_passed < 10:
        return "Report Received"
    elif minutes_passed < 20:
        return "Work Order Dispatched"
    else:
        return "Issue Resolved"


# ============================
#  AI ANALYSIS
# ============================

def analyze_image_unified(image_bytes: bytes) -> dict:
    """
    Run the ML model (HOG + SVM) on the uploaded image.
    """
    if MODEL is None:
        return {
            "issue_type_ai": "Other Urban Issue",
            "ai_severity": "Low",
            "confidence": 0.5
        }

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "issue_type_ai": "Other Urban Issue",
            "ai_severity": "Low",
            "confidence": 0.5
        }

    # Preprocess
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG features
    hog_features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    hog_features = hog_features.reshape(1, -1)

    # Predict
    prediction = MODEL.predict(hog_features)[0]      
    confidence = float(max(MODEL.predict_proba(hog_features)[0]))

    # Severity based on confidence
    if confidence > 0.8:
        severity = "High"
    elif confidence > 0.5:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "issue_type_ai": prediction.replace("_", " "),  
        "ai_severity": severity,
        "confidence": confidence
    }


def calculate_priority_score_unified(ai_data, existing_reports_count):
    severity_map = {"High": 1.0, "Medium": 0.6, "Low": 0.3}
    sev = severity_map.get(ai_data.get("ai_severity", "Low"), 0.3)
    dens = min(existing_reports_count / 5.0, 1.0)
    conf = ai_data.get("confidence", 0.5)

    score = (sev * 5) + (dens * 3) + (conf * 2)
    return round(score, 2)


# ============================
#  SERIALIZER
# ============================

def serialize_report(report):
    """
    Convert MongoDB document into JSON for frontend with dynamic status.
    """
    report["_id"] = str(report["_id"])
    lon, lat = report["location"]["coordinates"]

    return {
        "id": report["_id"],
        "issue_type": report["issue_type"],
        "description": report["description"],
        # Status is calculated dynamically every time the dashboard is loaded
        "status": get_real_world_status(report["created_at"]),
        "priority_score": report["ai_priority_score"],
        "target_authority": AUTHORITY_MAP.get(report["issue_type"], "GHMC"),
        "image_url": f"/{report['image_path']}",
        "location": {"lon": lon, "lat": lat},
        "created_at": report["created_at"].isoformat()
    }


# ============================
#  ROUTES
# ============================

def register_api_endpoints(app):

    @app.route("/api/report", methods=["POST"])
    def submit_report():
        data = request.form
        reports_collection = mongo.db.reports

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file: FileStorage = request.files["image"]

        # Read image for AI
        image_bytes = file.read()
        file.stream.seek(0)

        ai_data = analyze_image_unified(image_bytes)

        # Normalize AI label & combine with user input
        raw_label = ai_data["issue_type_ai"]
        normalized_ai = raw_label.strip().title()

        user_issue = data.get("issue_type")
        issue_type_final = user_issue.strip() if user_issue else normalized_ai

        target_authority = AUTHORITY_MAP.get(issue_type_final, "GHMC")

        try:
            lon = float(data.get("longitude"))
            lat = float(data.get('latitude'))
        except Exception:
            return jsonify({"error": "Invalid coordinates"}), 400

        # Geo Query (50m radius)
        RADIUS_IN_RADIANS = 50 / 6378137.0
        geojson_location = {"type": "Point", "coordinates": [lon, lat]}

        existing_reports = reports_collection.count_documents({
            "location": {
                "$geoWithin": {
                    "$centerSphere": [[lon, lat], RADIUS_IN_RADIANS]
                }
            }
        })

        priority_score = calculate_priority_score_unified(ai_data, existing_reports)

        # Save Image locally
        authority_folder = os.path.join(BASE_UPLOAD_FOLDER, target_authority)
        os.makedirs(authority_folder, exist_ok=True)

        filename = secure_filename(
            f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        )
        image_path = os.path.join(authority_folder, filename)
        file.save(image_path)

        # Save to DB
        doc = {
            "issue_type": issue_type_final,
            "description": data.get("description", "No description provided."),
            "image_path": image_path,
            "location": geojson_location,
            "ai_priority_score": priority_score,
            "ai_severity": ai_data["ai_severity"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        result = reports_collection.insert_one(doc)

        return jsonify({
            "message": "Report submitted successfully.",
            "report_id": str(result.inserted_id),
            "ai_priority": priority_score,
            "target_authority": target_authority
        }), 201

    @app.route("/api/reports", methods=["GET"])
    def get_all_reports():
        reports_collection = mongo.db.reports
        authority_filter = request.args.get("authority")
        status_filter = request.args.get("status")

        query = {}
        if authority_filter:
            issues = [i for i, a in AUTHORITY_MAP.items() if a == authority_filter]
            if issues:
                query["issue_type"] = {"$in": issues}

        # Fetch all matching reports
        cursor = reports_collection.find(query).sort("created_at", -1)
        
        # Note: Dynamic status is applied during serialization
        return jsonify([serialize_report(r) for r in cursor])


# ============================
#  APP FACTORY
# ============================

def create_app(config_dict=None):
    app = Flask(__name__, static_url_path="/static", static_folder="static")

    app.config.from_object(Config)
    if config_dict:
        app.config.update(config_dict)

    CORS(app)
    mongo.init_app(app)

    register_api_endpoints(app)

    with app.app_context():
        try:
            mongo.db.reports.create_index([("location", "2dsphere")])
        except Exception:
            pass

    return app
