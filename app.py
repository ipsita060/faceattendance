import os
import io
import base64
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
CSV_PATH = os.path.join(BASE_DIR, "attendance.csv")

# Ensure attendance.csv exists
if not os.path.exists(CSV_PATH):
    df_init = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
    df_init.to_csv(CSV_PATH, index=False)

@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "index.html"))

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(BASE_DIR, path)

@app.route("/api/status", methods=["GET"])
def get_status():
    registered = [
        os.path.splitext(f)[0].strip().title()
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    return jsonify({
        "status": "online",
        "registered_count": len(registered),
        "registered_users": registered,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/attendance", methods=["GET", "POST"])
def attendance_api():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name", "").strip().title()
        if not name:
            return jsonify({"error": "Name is required"}), 400

        now = datetime.now()
        date_str = data.get("date", now.strftime("%Y-%m-%d"))
        time_str = data.get("time", now.strftime("%H:%M:%S"))
        status = data.get("status", "Present")

        try:
            df = pd.read_csv(CSV_PATH)
        except Exception:
            df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

        # Check duplicate for the same date
        existing = df[(df["Name"].str.upper() == name.upper()) & (df["Date"] == date_str)]
        if not existing.empty:
            return jsonify({
                "message": f"Attendance already recorded today for {name} at {existing.iloc[0]['Time']}",
                "recorded": False,
                "record": existing.iloc[0].to_dict()
            })

        new_entry = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str, "Status": status}])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

        return jsonify({
            "message": f"Attendance marked for {name}",
            "recorded": True,
            "record": {"Name": name, "Date": date_str, "Time": time_str, "Status": status}
        })

    # GET: return list
    try:
        df = pd.read_csv(CSV_PATH)
        # Ensure column compatibility
        if "Date" not in df.columns:
            df["Date"] = datetime.now().strftime("%Y-%m-%d")
        if "Status" not in df.columns:
            df["Status"] = "Present"
        records = df.to_dict(orient="records")
        return jsonify({"records": records, "total": len(records)})
    except Exception as e:
        return jsonify({"error": str(e), "records": []}), 500

@app.route("/api/attendance/clear", methods=["POST"])
def clear_attendance():
    df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
    df.to_csv(CSV_PATH, index=False)
    return jsonify({"message": "Attendance log cleared successfully."})

@app.route("/api/attendance/export", methods=["GET"])
def export_csv():
    if os.path.exists(CSV_PATH):
        return send_file(CSV_PATH, as_attachment=True, download_name="attendance_report.csv", mimetype="text/csv")
    return jsonify({"error": "CSV file not found"}), 404

@app.route("/api/register", methods=["POST"])
def register_face():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    image_b64 = data.get("image", "")

    if not name or not image_b64:
        return jsonify({"error": "Name and base64 image data are required"}), 400

    try:
        # Strip header if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(image_b64)
        
        safe_name = "".join([c for c in name if c.isalnum() or c in (" ", "_")]).strip().lower().replace(" ", "_")
        file_path = os.path.join(IMAGES_DIR, f"{safe_name}.jpeg")
        
        with open(file_path, "wb") as f:
            f.write(img_bytes)

        return jsonify({
            "message": f"Successfully registered {name}",
            "name": name.title(),
            "image_url": f"images/{safe_name}.jpeg"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Face Attendance Web Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
