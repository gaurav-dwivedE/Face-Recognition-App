import os
import time
import base64

import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/face_recognition")
DB_NAME = os.getenv("MONGO_DB", "face_recognition")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "users")
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", "5"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.6"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
REPO_ROOT = os.getenv("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__)

_cached_faces = {
    "loaded_at": 0,
    "embeddings": [],
    "user_ids": []
}


def _abs_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(REPO_ROOT, path_str)


def load_known_faces(force=False):
    global _cached_faces

    now = time.time()
    if not force and (now - _cached_faces["loaded_at"]) < CACHE_TTL_SECONDS:
        return _cached_faces

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    users = list(db[COLLECTION_NAME].find({}))

    embeddings = []
    user_ids = []

    for user in users:
        image_path = user.get("imagePath")
        if not image_path:
            continue

        abs_path = _abs_path(image_path)
        if not os.path.exists(abs_path):
            continue

        image = face_recognition.load_image_file(abs_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            continue

        embeddings.append(encodings[0])
        user_ids.append(str(user.get("_id")))

    client.close()

    _cached_faces = {
        "loaded_at": now,
        "embeddings": embeddings,
        "user_ids": user_ids
    }

    return _cached_faces


def frame_to_base64(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def match_faces_in_frame(frame, known_embeddings, known_user_ids):
    if len(known_embeddings) == 0:
        return []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    results = {}

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_embeddings, face_encoding)
        if len(distances) == 0:
            continue

        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance > MATCH_THRESHOLD:
            continue

        user_id = known_user_ids[best_index]
        confidence = max(0.0, min(1.0, 1.0 - best_distance))
        face_crop = frame[top:bottom, left:right]
        snapshot = frame_to_base64(face_crop) if face_crop.size else None

        existing = results.get(user_id)
        if not existing or confidence > existing["confidence"]:
            results[user_id] = {
                "userId": user_id,
                "confidence": confidence,
                "snapshot": snapshot,
                "bbox": {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left)
                }
            }

    return list(results.values())


def recognize_faces(video_path):
    cache = load_known_faces()
    known_embeddings = cache["embeddings"]
    known_user_ids = cache["user_ids"]

    if not known_embeddings:
        return []

    cap = cv2.VideoCapture(video_path)
    aggregate = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            continue

        frame_matches = match_faces_in_frame(frame, known_embeddings, known_user_ids)
        for match in frame_matches:
            user_id = match["userId"]
            existing = aggregate.get(user_id)
            if not existing or match["confidence"] > existing["confidence"]:
                aggregate[user_id] = match

        frame_idx += 1

    cap.release()
    return list(aggregate.values())


def recognize_single_frame(frame):
    cache = load_known_faces()
    known_embeddings = cache["embeddings"]
    known_user_ids = cache["user_ids"]
    return match_faces_in_frame(frame, known_embeddings, known_user_ids)


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "python-ai"})


@app.post("/reload-known-faces")
def reload_known_faces():
    cache = load_known_faces(force=True)
    return jsonify({"ok": True, "knownFaces": len(cache["embeddings"])})


@app.post("/recognize")
def recognize():
    if "video" not in request.files:
        return jsonify({"error": "Missing video file"}), 400

    video = request.files["video"]
    temp_path = os.path.join("/tmp", f"video-{int(time.time())}.mp4")
    video.save(temp_path)

    try:
        matches = recognize_faces(temp_path)
        return jsonify({"matches": matches})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/recognize-frame")
def recognize_frame():
    if "frame" not in request.files:
        return jsonify({"error": "Missing frame image"}), 400

    frame_file = request.files["frame"]
    raw = frame_file.read()
    if not raw:
        return jsonify({"error": "Empty frame image"}), 400

    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid frame image"}), 400

    matches = recognize_single_frame(frame)
    return jsonify({"matches": matches})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
