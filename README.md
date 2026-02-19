# Urban Issues Crowd-Sourced Reporting Platform

## Overview
This project is a full-stack web application for crowd-sourced reporting and management of urban issues (e.g., potholes, garbage, streetlights, waterlogging, unsafe areas). It features:
- User authentication (citizen/admin roles)
- AI-powered image classification for issue type
- MongoDB database for storing reports and user data
- Admin and citizen dashboards
- File uploads and image storage
- Notification system

---

## Project Structure

```
.
├── app.py                  # Main Flask application (API, routes, logic)
├── models.py               # MongoDB connection and models
├── config.py               # Configuration (secrets, DB URI, etc.)
├── requirements.txt        # Python dependencies
├── run.py                  # App entry point
├── train_model.py          # Script to train the ML model
├── street_issue_model.pkl  # Trained ML model (HOG + SVM)
├── admin.html              # Admin dashboard UI
├── citizen.html            # Citizen dashboard UI
├── index.html              # Main dashboard (role-based)
├── login.html              # Login page
├── static/
│   ├── dashboard.css       # Main CSS for UI
│   └── uploads/            # Uploaded images (by authority)
├── start.bat               # Windows batch file to start the app
├── .env                    # Environment variables (not committed)
├── .gitignore              # Git ignore rules
└── ...
```

---

## File Descriptions & Relationships

### Core Backend
- **app.py**: Main Flask app. Handles all API endpoints (auth, report submission, AI analysis, notifications). Loads the ML model and serves HTML pages. Imports `models.py` and `config.py`.
- **models.py**: Sets up MongoDB connection (via Flask-PyMongo). Used by `app.py` for all DB operations.
- **config.py**: Contains configuration (secret keys, DB URIs). Imported by `app.py`.
- **requirements.txt**: Lists all Python dependencies.
- **run.py**: Entry point to start the Flask app (may import and run `app.py`).

### Machine Learning
- **train_model.py**: Script to train the ML model (HOG + SVM) for classifying urban issues from images. Outputs `street_issue_model.pkl`.
- **street_issue_model.pkl**: The trained ML model loaded by `app.py` for AI analysis.

### Frontend
- **index.html**: Main dashboard for users (citizens/admins). Contains forms for submitting reports, viewing issues, and role-based UI. Uses JavaScript to call Flask API endpoints.
- **login.html**: Login page for users.
- **admin.html**: Admin dashboard (served to users with admin role).
- **citizen.html**: Citizen dashboard (served to regular users).

### Static Files
- **static/dashboard.css**: Main CSS for styling the dashboard and other pages.
- **static/uploads/**: Stores uploaded images for reports, organized by authority (GHMC, HMWSSB, TSSPDCL).

### Other
- **start.bat**: Batch file to start the application on Windows.
- **.env**: Environment variable file (for secrets, DB URIs, etc.).
- **.gitignore**: Specifies files/folders to ignore in git.

---

## How the System Works

1. **User Authentication**
   - Users sign up or log in (citizen or admin role).
   - Auth endpoints in `app.py` manage sessions and roles.

2. **Report Submission**
   - Citizens submit reports via the dashboard (`index.html`).
   - Each report includes an image, description, and location.
   - The image is analyzed by the ML model (`street_issue_model.pkl`) to classify the issue type.
   - Reports are stored in MongoDB (via `models.py`).
   - Images are saved in `static/uploads/` under the relevant authority.

3. **Dashboards**
   - Citizens see their submitted reports and can submit new ones.
   - Admins see all reports and can manage them.
   - Role-based dashboards are served by Flask (`admin.html`, `citizen.html`, `index.html`).

4. **Notifications**
   - Users receive notifications about report status and updates.

5. **AI Model**
   - `train_model.py` is used to train the ML model on labeled images.
   - The trained model is loaded by `app.py` for real-time image classification.

---

## Setup & Deployment

1. **Clone the repository**
2. **Create a virtual environment and install dependencies:**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Set up your `.env` file** with the required environment variables (see `.env.example` if available).
4. **Start MongoDB** (local or remote, as configured).
5. **Run the application:**
   ```sh
   python run.py
   ```
6. **Access the app** at [http://localhost:5000](http://localhost:5000)

---

## Notes
- For production, remove development/test scripts and cache files (already cleaned).
- Only keep essential files as described above.
- Make sure to secure your `.env` and never commit secrets.

---

## License
Specify your license here.

---

## Contact
For questions or support, contact the project maintainer.
