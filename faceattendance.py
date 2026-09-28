import os
import sys
import argparse
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import face_recognition

# ------------------ ARGUMENTS & CONFIG ------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Smart AI Face Recognition Attendance System")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Face matching tolerance threshold (lower is stricter, default: 0.5)")
    parser.add_argument("--image_dir", type=str, default="", help="Directory containing registered user portraits")
    parser.add_argument("--csv", type=str, default="attendance.csv", help="CSV file path to store attendance")
    parser.add_argument("--allow_multiple", action="store_true", help="Allow marking attendance multiple times per day")
    return parser.parse_args()

# ------------------ LOCATE IMAGES DIRECTORY ------------------
def get_images_directory(custom_path=""):
    if custom_path and os.path.exists(custom_path):
        return custom_path
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "images"),
        os.path.join(base_dir, "images "),
        os.path.join(base_dir, "..", "images"),
        "images",
        "images "
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    # Fallback to creating images dir
    default_dir = os.path.join(base_dir, "images")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir

# ------------------ ENCODING REFERENCE FACES ------------------
def load_and_encode_faces(images_dir):
    images = []
    class_names = []
    encode_list = []

    print(f"[*] Scanning registered personnel portraits from: {images_dir}")
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")

    for file_name in sorted(os.listdir(images_dir)):
        if file_name.lower().endswith(valid_exts):
            img_path = os.path.join(images_dir, file_name)
            cur_img = cv2.imread(img_path)
            if cur_img is not None:
                # Clean up name: remove extension and trailing/leading spaces
                name = os.path.splitext(file_name)[0].strip().title()
                rgb_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_img)
                if len(encs) > 0:
                    encode_list.append(encs[0])
                    class_names.append(name)
                    images.append(cur_img)
                    print(f"  [+] Encoded: {name} (128-d descriptor)")
                else:
                    print(f"  [!] Warning: No face detected in {file_name}")

    print(f"[*] Successfully encoded {len(encode_list)} registered users: {', '.join(class_names)}")
    return encode_list, class_names

# ------------------ ATTENDANCE LOGGING ------------------
def mark_attendance(name, csv_path="attendance.csv", allow_multiple=False):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Initialize CSV if missing
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv(csv_path, index=False)

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

    # Ensure required columns exist
    if "Date" not in df.columns:
        df["Date"] = date_str
    if "Status" not in df.columns:
        df["Status"] = "Present"

    # Check duplicate for today
    if not allow_multiple:
        existing_today = df[(df["Name"].str.upper() == name.upper()) & (df["Date"] == date_str)]
        if not existing_today.empty:
            return False, f"Already recorded today at {existing_today.iloc[0]['Time']}"

    new_row = pd.DataFrame([{
        "Name": name,
        "Date": date_str,
        "Time": time_str,
        "Status": "Present"
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[ATTENDANCE MARKED] >> {name} | Date: {date_str} | Time: {time_str}")
    return True, f"Attendance marked at {time_str}"

# ------------------ MAIN REAL-TIME LOOP ------------------
def run_attendance_system():
    args = parse_arguments()
    images_dir = get_images_directory(args.image_dir)
    encode_list_known, class_names = load_and_encode_faces(images_dir)

    if not encode_list_known:
        print("[!] No face encodings available. Please add portrait images to the 'images/' folder.")
        sys.exit(1)

    print(f"[*] Initializing camera index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[!] Camera {args.camera} not accessible.")
        print("[*] Tip: Check camera permissions or try --camera 1")
        sys.exit(1)

    print("\n" + "="*50)
    print("   AI FACE RECOGNITION ATTENDANCE ACTIVE")
    print("   Press 'q' or 'ESC' or 'ENTER' to exit")
    print("="*50 + "\n")

    last_announcement = {}
    fps_history = []
    prev_time = cv2.getTickCount()

    while True:
        success, img = cap.read()
        if not success:
            print("[!] Failed to read frame from camera stream.")
            break

        # Calculate FPS
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Downscale for high-speed processing
        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        faces_cur_frame = face_recognition.face_locations(img_small_rgb)
        encodes_cur_frame = face_recognition.face_encodings(img_small_rgb, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encode_list_known, encode_face, tolerance=args.tolerance)
            face_dis = face_recognition.face_distance(encode_list_known, encode_face)

            y1, x2, y2, x1 = [coord * 4 for coord in face_loc]

            if len(face_dis) > 0:
                match_index = np.argmin(face_dis)
                confidence = max(0.0, min(100.0, (1.0 - face_dis[match_index]) * 100))

                if matches[match_index]:
                    name = class_names[match_index]
                    
                    # Mark attendance
                    now_ts = datetime.now().timestamp()
                    if name not in last_announcement or (now_ts - last_announcement[name] > 5):
                        last_announcement[name] = now_ts
                        mark_attendance(name, args.csv, args.allow_multiple)

                    # Green box & modern label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (46, 204, 113), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (46, 204, 113), cv2.FILLED)
                    cv2.putText(img, f"{name} ({confidence:.0f}%)", (x1 + 6, y2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                else:
                    # Red box - Not Registered
                    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 235), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (60, 60, 235), cv2.FILLED)
                    cv2.putText(img, "UNKNOWN VISITOR", (x1 + 6, y2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        # Top status bar overlay
        cv2.rectangle(img, (0, 0), (img.shape[1], 45), (20, 24, 33), cv2.FILLED)
        cv2.putText(img, f"Face Attendance AI | FPS: {avg_fps:.1f} | Registered: {len(class_names)}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 180), 2)

        cv2.imshow("Smart Face Recognition Attendance System", img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27, 13):  # 'q', ESC, or ENTER
            print("\n[*] Exiting Face Recognition Attendance.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attendance_system()
