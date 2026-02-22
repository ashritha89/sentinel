from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from config import Config
from models import mongo
from datetime import datetime, timedelta
import os
import logging
import sys
# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger('pymongo').setLevel(logging.WARNING)
from werkzeug.utils import secure_filename
from flask_cors import CORS
import certifi
from werkzeug.datastructures import FileStorage
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import pickle
import cv2
from skimage.feature import hog
import numpy as np
import base64
from openai import OpenAI

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

def serialize_report(report, user_id=None, username=None):
    """
    Convert MongoDB document into JSON for frontend with stored status.
    """
    report_id = str(report["_id"])
    lon, lat = report["location"]["coordinates"]
    
    # Get upvote/verify data
    upvoted_by = report.get("upvoted_by", [])
    verified_by = report.get("verified_by", [])

    return {
        "_id": report_id,
        "id": report_id,
        "issue_type": report.get("issue_type", "Unknown"),
        "description": report.get("description", ""),
        "status": report.get("status", "Pending"),
        "priority_score": report.get("ai_priority_score", 5),
        "target_authority": AUTHORITY_MAP.get(report.get("issue_type", ""), "GHMC"),
        "image_url": f"/{report.get('image_path', 'static/placeholder.jpg')}",
        "latitude": lat,
        "longitude": lon,
        "location": {"lon": lon, "lat": lat},
        "created_at": report.get("created_at", datetime.utcnow()).isoformat(),
        "upvote_count": report.get("upvote_count", 0),
        "verified_count": report.get("verified_count", 0),
        "has_upvoted": user_id in upvoted_by if user_id else False,
        "has_verified": user_id in verified_by if user_id else False,
        "is_mine": report.get("submitted_by") == username if username else False
    }


# ============================
#  AUTH DECORATOR
# ============================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# ============================
#  ROUTES
# ============================

def register_api_endpoints(app):

    # Public route for login page
    @app.route("/login")
    def login_page():
        if 'user_id' in session:
            return redirect(url_for('serve_index'))
        return send_from_directory(BASE_DIR, 'login.html')

    # Protected Dashboard
    @app.route("/")
    @login_required
    def serve_index():
        # Check user role and serve appropriate dashboard
        user_role = session.get('role', 'citizen')
        if user_role == 'admin':
            return send_from_directory(BASE_DIR, 'admin.html')
        else:
            return send_from_directory(BASE_DIR, 'citizen.html')
    
    # Admin dashboard route
    @app.route("/admin")
    @login_required
    def serve_admin():
        if session.get('role') != 'admin':
            return redirect(url_for('serve_index'))
        return send_from_directory(BASE_DIR, 'admin.html')
    
    @app.route("/api/auth/signup", methods=["POST"])
    def signup():
        try:
            data = request.json
            username = data.get("username")
            email = data.get("email")
            password = data.get("password")
            role = data.get("role", "citizen") # citizen or admin

            if not username or not password or not email:
                return jsonify({"error": "Missing fields"}), 400
            
            users = mongo.db.users
            existing_user = users.find_one({"$or": [{"username": username}, {"email": email}]})
            if existing_user:
                return jsonify({"error": "User or Email already exists"}), 400

            hashed_pw = generate_password_hash(password)
            users.insert_one({
                "username": username,
                "email": email,
                "password": hashed_pw,
                "role": role,
                "created_at": datetime.utcnow()
            })
            logging.info(f"New user created: {username}")
            return jsonify({"message": "User created successfully"}), 201
        except Exception as e:
            logging.error(f"Signup error: {str(e)}")
            return jsonify({"error": "Database connection error. Please ensure MongoDB is running."}), 500

    @app.route("/api/auth/login", methods=["POST"])
    def login():
        try:
            data = request.json
            email = data.get("email")
            password = data.get("password")

            if not email or not password:
                return jsonify({"error": "Email and password are required"}), 400

            # Allow login by email only as requested, or we could do username too.
            # Let's assume Email based on the prompt "keep mail id in login page".
            if mongo.db is None:
                raise Exception("MongoDB not initialized")

            user = mongo.db.users.find_one({"email": email})
            
            if user and check_password_hash(user["password"], password):
                session['user_id'] = str(user["_id"])
                session['username'] = user["username"]
                session['email'] = user.get("email")
                session['role'] = user["role"]
                logging.info(f"User logged in: {email}")
                return jsonify({"message": "Login successful", "role": user["role"]}), 200
            
            return jsonify({"error": "Invalid email or password"}), 401
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            return jsonify({"error": "Database connection error. Please ensure MongoDB is running."}), 500

    @app.route("/api/auth/logout")
    def logout():
        session.clear()
        return redirect(url_for('login_page'))
    
    @app.route("/api/auth/me")
    def get_current_user():
        if 'user_id' not in session:
            return jsonify({"error": "Not logged in"}), 401
        return jsonify({
            "username": session['username'],
            "role": session['role']
        })

    @app.route("/api/analyze-image", methods=["POST"])
    @login_required
    def analyze_image_groq():
        """Analyze an image using Groq Vision API to detect issue type"""
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        
        try:
            # Read and encode image to base64
            image_bytes = file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine image type
            content_type = file.content_type or 'image/jpeg'
            
            # Initialize Groq client (uses OpenAI SDK with custom base URL)
            groq_key = os.environ.get('GROQ_API_KEY')
            if not groq_key:
                return jsonify({"error": "Groq API key not configured"}), 500
            
            client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
            # Call Groq Vision API with llama-4-scout
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this image and classify it into ONE of these urban issue categories:
- Pothole (road damage, cracks, potholes)
- Streetlight (broken or non-functional street lights)
- Garbage (overflowing bins, litter, waste)
- Waterlogging (flooded areas, drainage issues)
- Unsafe Area (encroachments, dangerous structures)

Respond in JSON format only:
{"issue_type": "category", "confidence": 0.0-1.0, "description": "brief description of what you see"}"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{content_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            
            # Try to parse JSON from response
            import json
            try:
                # Clean up response if it has markdown code blocks
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                
                result = json.loads(result_text.strip())
                return jsonify(result)
            except json.JSONDecodeError:
                # If JSON parsing fails, return a basic response
                return jsonify({
                    "issue_type": "Unknown",
                    "confidence": 0.5,
                    "description": result_text
                })
                
        except Exception as e:
            logging.error(f"Groq analysis error: {str(e)}")
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


    @app.route("/api/report", methods=["POST"])
    @login_required
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

        # Geo Query (50m radius) - Enhanced Duplicate Detection
        RADIUS_IN_RADIANS = 50 / 6378137.0
        geojson_location = {"type": "Point", "coordinates": [lon, lat]}

        # Check for duplicate: same issue type within 50m, not completed
        duplicate_report = reports_collection.find_one({
            "issue_type": issue_type_final,
            "status": {"$ne": "Completed"},
            "location": {
                "$geoWithin": {
                    "$centerSphere": [[lon, lat], RADIUS_IN_RADIANS]
                }
            }
        })

        if duplicate_report:
            # Merge with existing report - increment report count and add upvote
            reports_collection.update_one(
                {"_id": duplicate_report["_id"]},
                {
                    "$inc": {"report_count": 1, "upvote_count": 1},
                    "$addToSet": {"upvoted_by": session.get('user_id')},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            return jsonify({
                "message": "Similar issue already reported nearby! Your report has been merged and upvoted.",
                "is_duplicate": True,
                "original_report_id": str(duplicate_report["_id"]),
                "report_count": duplicate_report.get("report_count", 1) + 1,
                "target_authority": AUTHORITY_MAP.get(duplicate_report.get("issue_type", ""), "GHMC")
            }), 200

        # Count all nearby reports for priority calculation
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
            "updated_at": datetime.utcnow(),
            "submitted_by": session.get('username', 'Anonymous'),
            "status": "Pending",
            "report_count": 1,
            "upvote_count": 0,
            "upvoted_by": [],
            "verified_count": 0,
            "verified_by": []
        }

        result = reports_collection.insert_one(doc)

        return jsonify({
            "message": "Report submitted successfully.",
            "report_id": str(result.inserted_id),
            "ai_priority": priority_score,
            "target_authority": target_authority,
            "is_duplicate": False
        }), 201

    @app.route("/api/reports", methods=["GET"])
    @login_required
    def get_all_reports():
        try:
            reports_collection = mongo.db.reports
            authority_filter = request.args.get("authority")
            status_filter = request.args.get("status")
            user_id = session.get('user_id')

            query = {}
            if authority_filter:
                issues = [i for i, a in AUTHORITY_MAP.items() if a == authority_filter]
                if issues:
                    query["issue_type"] = {"$in": issues}

            # Fetch all matching reports
            cursor = reports_collection.find(query).sort("created_at", -1)
            
            username = session.get('username')
            
            reports = []
            for r in cursor:
                try:
                    reports.append(serialize_report(r, user_id, username))
                except Exception as e:
                    logging.error(f"Error serializing report {r.get('_id')}: {str(e)}")
                    continue
            
            logging.info(f"Returning {len(reports)} reports")
            return jsonify(reports)
        except Exception as e:
            logging.error(f"Error fetching reports: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/reports/<report_id>/status", methods=["PUT"])
    @login_required
    def update_report_status(report_id):
        """Update the status of a report (admin only)"""
        # Check if user is admin
        if session.get('role') != 'admin':
            return jsonify({"error": "Unauthorized. Admin access required."}), 403
        
        data = request.json
        new_status = data.get("status")
        
        # Validate status
        valid_statuses = ["Pending", "Accepted", "Assigned", "Completed"]
        if new_status not in valid_statuses:
            return jsonify({"error": f"Invalid status. Must be one of: {valid_statuses}"}), 400
        
        try:
            from bson import ObjectId
            reports_collection = mongo.db.reports
            
            # Update the report status
            result = reports_collection.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": {
                    "status": new_status,
                    "updated_at": datetime.utcnow(),
                    "updated_by": session.get('username')
                }}
            )
            
            if result.matched_count == 0:
                return jsonify({"error": "Report not found"}), 404
            
            # Create notification for the report owner
            report = reports_collection.find_one({"_id": ObjectId(report_id)})
            if report and report.get("submitted_by"):
                status_messages = {
                    "Accepted": "Your report has been accepted by the authority! 🎉",
                    "Assigned": "A worker has been assigned to your issue! 👷",
                    "Completed": "Great news! Your reported issue has been resolved! ✅"
                }
                if new_status in status_messages:
                    notifications_collection = mongo.db.notifications
                    notifications_collection.insert_one({
                        "user_id": report["submitted_by"],
                        "type": "status_update",
                        "message": status_messages[new_status],
                        "report_id": report_id,
                        "issue_type": report.get("issue_type", "Unknown"),
                        "new_status": new_status,
                        "read": False,
                        "created_at": datetime.utcnow()
                    })
            
            logging.info(f"Report {report_id} status updated to {new_status} by {session.get('username')}")
            return jsonify({"message": "Status updated successfully", "status": new_status}), 200
            
        except Exception as e:
            logging.error(f"Error updating status: {str(e)}")
            return jsonify({"error": "Failed to update status"}), 500

    @app.route("/api/reports/<report_id>/upvote", methods=["POST"])
    @login_required
    def upvote_report(report_id):
        """Upvote a report to increase its priority"""
        try:
            from bson import ObjectId
            reports_collection = mongo.db.reports
            user_id = session.get('user_id')
            
            # Check if already upvoted
            report = reports_collection.find_one({"_id": ObjectId(report_id)})
            if not report:
                return jsonify({"error": "Report not found"}), 404
            
            upvoted_by = report.get("upvoted_by", [])
            
            if user_id in upvoted_by:
                # Remove upvote
                reports_collection.update_one(
                    {"_id": ObjectId(report_id)},
                    {
                        "$pull": {"upvoted_by": user_id},
                        "$inc": {"upvote_count": -1}
                    }
                )
                return jsonify({"message": "Upvote removed", "upvoted": False}), 200
            else:
                # Add upvote
                reports_collection.update_one(
                    {"_id": ObjectId(report_id)},
                    {
                        "$addToSet": {"upvoted_by": user_id},
                        "$inc": {"upvote_count": 1}
                    }
                )
                return jsonify({"message": "Upvoted!", "upvoted": True}), 200
                
        except Exception as e:
            logging.error(f"Error upvoting: {str(e)}")
            return jsonify({"error": "Failed to upvote"}), 500

    @app.route("/api/reports/<report_id>/verify", methods=["POST"])
    @login_required
    def verify_report(report_id):
        """Verify that an issue is still unresolved"""
        try:
            from bson import ObjectId
            reports_collection = mongo.db.reports
            user_id = session.get('user_id')
            
            report = reports_collection.find_one({"_id": ObjectId(report_id)})
            if not report:
                return jsonify({"error": "Report not found"}), 404
            
            verified_by = report.get("verified_by", [])
            
            if user_id in verified_by:
                return jsonify({"message": "Already verified by you", "verified": True}), 200
            
            # Add verification
            reports_collection.update_one(
                {"_id": ObjectId(report_id)},
                {
                    "$addToSet": {"verified_by": user_id},
                    "$inc": {"verified_count": 1},
                    "$set": {"last_verified_at": datetime.utcnow()}
                }
            )
            return jsonify({"message": "Verified as unresolved!", "verified": True}), 200
                
        except Exception as e:
            logging.error(f"Error verifying: {str(e)}")
            return jsonify({"error": "Failed to verify"}), 500

    @app.route("/api/reports/<report_id>", methods=["GET"])
    @login_required
    def get_single_report(report_id):
        """Get a single report by ID"""
        try:
            from bson import ObjectId
            reports_collection = mongo.db.reports
            report = reports_collection.find_one({"_id": ObjectId(report_id)})
            
            if not report:
                return jsonify({"error": "Report not found"}), 404
            
            return jsonify(serialize_report(report, session.get('user_id'), session.get('username')))
        except Exception as e:
            logging.error(f"Error fetching report: {str(e)}")
            return jsonify({"error": "Failed to fetch report"}), 500

    # ============================
    #  NOTIFICATIONS SYSTEM
    # ============================
    
    @app.route("/api/notifications", methods=["GET"])
    @login_required
    def get_notifications():
        """Get notifications for the current user"""
        try:
            from bson import ObjectId
            notifications_collection = mongo.db.notifications
            user_id = session.get('user_id')
            
            notifications = list(notifications_collection.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(20))
            
            for n in notifications:
                n["_id"] = str(n["_id"])
                n["created_at"] = n["created_at"].isoformat() if n.get("created_at") else None
            
            return jsonify(notifications)
        except Exception as e:
            logging.error(f"Error fetching notifications: {str(e)}")
            return jsonify({"error": "Failed to fetch notifications"}), 500

    @app.route("/api/notifications/unread-count", methods=["GET"])
    @login_required
    def get_unread_count():
        """Get count of unread notifications"""
        try:
            notifications_collection = mongo.db.notifications
            user_id = session.get('user_id')
            count = notifications_collection.count_documents({"user_id": user_id, "read": False})
            return jsonify({"count": count})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/notifications/<notification_id>/read", methods=["PUT"])
    @login_required
    def mark_notification_read(notification_id):
        """Mark a notification as read"""
        try:
            from bson import ObjectId
            notifications_collection = mongo.db.notifications
            notifications_collection.update_one(
                {"_id": ObjectId(notification_id)},
                {"$set": {"read": True}}
            )
            return jsonify({"message": "Marked as read"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/notifications/read-all", methods=["PUT"])
    @login_required
    def mark_all_notifications_read():
        """Mark all notifications as read"""
        try:
            notifications_collection = mongo.db.notifications
            user_id = session.get('user_id')
            notifications_collection.update_many(
                {"user_id": user_id},
                {"$set": {"read": True}}
            )
            return jsonify({"message": "All notifications marked as read"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def create_notification(user_id, notification_type, message, report_id=None):
        """Helper function to create a notification"""
        try:
            notifications_collection = mongo.db.notifications
            notifications_collection.insert_one({
                "user_id": user_id,
                "type": notification_type,
                "message": message,
                "report_id": str(report_id) if report_id else None,
                "read": False,
                "created_at": datetime.utcnow()
            })
        except Exception as e:
            logging.error(f"Failed to create notification: {str(e)}")


# ============================
#  APP FACTORY
# ============================

def create_app(config_dict=None):
    app = Flask(__name__, static_url_path="/static", static_folder="static")

    app.config.from_object(Config)
    if config_dict:
        app.config.update(config_dict)

    CORS(app)
    mongo.init_app(app, tlsCAFile=certifi.where())

    register_api_endpoints(app)

    with app.app_context():
        try:
            # Test MongoDB connection
            mongo.db.command('ping')
            logging.info("✅ MongoDB connected successfully!")
            
            # Create indexes
            mongo.db.reports.create_index([("location", "2dsphere")])
            mongo.db.users.create_index("username", unique=True)
            logging.info("✅ Database indexes created")
        except Exception as e:
            logging.error("❌ MongoDB Connection Failed!")
            logging.error(f"Error: {str(e)}")
            logging.error("\n" + "="*60)
            logging.error("MONGODB NOT CONNECTED - Please do ONE of the following:")
            logging.error("="*60)
            logging.error("Option 1: Install MongoDB locally")
            logging.error("  - Download from: https://www.mongodb.com/try/download/community")
            logging.error("  - After install, run: net start MongoDB")
            logging.error("")
            logging.error("Option 2: Use MongoDB Atlas (Free Cloud)")
            logging.error("  - See MONGODB_ATLAS_SETUP.md for instructions")
            logging.error("  - Update MONGO_URI in .env file")
            logging.error("="*60 + "\n")

    return app

# For Gunicorn compatibility
app = create_app()
