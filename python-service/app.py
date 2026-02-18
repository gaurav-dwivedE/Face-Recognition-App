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
FRAME_INTERVAL = int(os.getenv("FRAME_INTERVAL", "1"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.50"))
MATCH_MARGIN = float(os.getenv("MATCH_MARGIN", "0.04"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "20"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
REPO_ROOT = os.getenv("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "hog")
PRIMARY_DETECTION_SCALE = float(os.getenv("PRIMARY_DETECTION_SCALE", "1.45"))
PRIMARY_UPSAMPLE = int(os.getenv("PRIMARY_UPSAMPLE", "1"))
SECONDARY_DETECTION_SCALE = float(os.getenv("SECONDARY_DETECTION_SCALE", "1.6"))
SECONDARY_UPSAMPLE = int(os.getenv("SECONDARY_UPSAMPLE", "1"))
SECONDARY_EVERY_N_FRAMES = max(1, int(os.getenv("SECONDARY_EVERY_N_FRAMES", "4")))
FAR_DETECTION_SCALE = float(os.getenv("FAR_DETECTION_SCALE", "1.9"))
FAR_UPSAMPLE = int(os.getenv("FAR_UPSAMPLE", "2"))
FAR_SCAN_ON_EMPTY = os.getenv("FAR_SCAN_ON_EMPTY", "1") == "1"
KNOWN_NUM_JITTERS = int(os.getenv("KNOWN_NUM_JITTERS", "1"))
FRAME_NUM_JITTERS = int(os.getenv("FRAME_NUM_JITTERS", "1"))
MAX_FRAME_SIDE = int(os.getenv("MAX_FRAME_SIDE", "1600"))
SNAPSHOT_MAX_SIDE = int(os.getenv("SNAPSHOT_MAX_SIDE", "96"))
SNAPSHOT_JPEG_QUALITY = int(os.getenv("SNAPSHOT_JPEG_QUALITY", "55"))
RETURN_MATCH_SNAPSHOTS = os.getenv("RETURN_MATCH_SNAPSHOTS", "1") == "1"
MATCH_SNAPSHOT_INTERVAL_MS = max(0, int(os.getenv("MATCH_SNAPSHOT_INTERVAL_MS", "1100")))
RETURN_UNKNOWN_SNAPSHOTS = os.getenv("RETURN_UNKNOWN_SNAPSHOTS", "0") == "1"

app = Flask(__name__)

_cached_faces = {
    "loaded_at": 0,
    "embeddings": [],
    "user_ids": []
}
_live_frame_counter = 0
_last_match_snapshot_at = {}


def _abs_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(REPO_ROOT, path_str)


def _valid_face_size(top, right, bottom, left):
    return (bottom - top) >= MIN_FACE_SIZE and (right - left) >= MIN_FACE_SIZE


def _resize_if_needed(frame):
    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side <= MAX_FRAME_SIDE:
        return frame
    scale = MAX_FRAME_SIDE / float(max_side)
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _scale_location_back(location, scale, max_h, max_w):
    top, right, bottom, left = location
    if scale <= 0:
        return (0, 0, 0, 0)

    top = int(round(top / scale))
    right = int(round(right / scale))
    bottom = int(round(bottom / scale))
    left = int(round(left / scale))

    top = max(0, min(top, max_h - 1))
    bottom = max(0, min(bottom, max_h))
    left = max(0, min(left, max_w - 1))
    right = max(0, min(right, max_w))

    return (top, right, bottom, left)


def _detect_faces_pass(rgb, scale, upsample, num_jitters):
    h, w = rgb.shape[:2]

    if scale != 1.0:
        detect_rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        detect_rgb = rgb

    locations_scaled = face_recognition.face_locations(
        detect_rgb,
        number_of_times_to_upsample=max(0, upsample),
        model=FACE_DETECTION_MODEL
    )

    if not locations_scaled:
        return [], []

    encodings = face_recognition.face_encodings(
        detect_rgb,
        known_face_locations=locations_scaled,
        num_jitters=max(1, num_jitters)
    )

    mapped_locations = [_scale_location_back(loc, scale, h, w) for loc in locations_scaled]
    return mapped_locations, encodings


def _iou(a, b):
    at, ar, ab, al = a
    bt, br, bb, bl = b

    inter_top = max(at, bt)
    inter_left = max(al, bl)
    inter_bottom = min(ab, bb)
    inter_right = min(ar, br)

    iw = max(0, inter_right - inter_left)
    ih = max(0, inter_bottom - inter_top)
    inter = iw * ih

    a_area = max(0, ar - al) * max(0, ab - at)
    b_area = max(0, br - bl) * max(0, bb - bt)

    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _merge_detections(primary_locations, primary_encodings, secondary_locations, secondary_encodings):
    merged_locations = list(primary_locations)
    merged_encodings = list(primary_encodings)

    for s_loc, s_enc in zip(secondary_locations, secondary_encodings):
        duplicate = False
        for p_loc in merged_locations:
            if _iou(s_loc, p_loc) > 0.45:
                duplicate = True
                break
        if not duplicate:
            merged_locations.append(s_loc)
            merged_encodings.append(s_enc)

    return merged_locations, merged_encodings


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
        locations = face_recognition.face_locations(
            image,
            number_of_times_to_upsample=max(0, PRIMARY_UPSAMPLE),
            model=FACE_DETECTION_MODEL
        )
        if not locations:
            continue

        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=locations,
            num_jitters=max(1, KNOWN_NUM_JITTERS)
        )
        if not encodings:
            continue

        for location, encoding in zip(locations, encodings):
            top, right, bottom, left = location
            if not _valid_face_size(top, right, bottom, left):
                continue
            embeddings.append(encoding)
            user_ids.append(str(user.get("_id")))

    client.close()

    _cached_faces = {
        "loaded_at": now,
        "embeddings": embeddings,
        "user_ids": user_ids
    }

    return _cached_faces


def frame_to_base64(frame):
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side > SNAPSHOT_MAX_SIDE:
        scale = SNAPSHOT_MAX_SIDE / float(max_side)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(20, min(95, SNAPSHOT_JPEG_QUALITY))]
    )
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def _is_ambiguous_match(distances):
    if len(distances) < 2:
        return False

    sorted_distances = np.sort(distances)
    best = float(sorted_distances[0])
    second_best = float(sorted_distances[1])
    return (second_best - best) < MATCH_MARGIN


def _append_unknown_face(unknown_faces, bbox, face_crop):
    payload = {"bbox": bbox}
    if RETURN_UNKNOWN_SNAPSHOTS:
        snapshot = frame_to_base64(face_crop)
        if snapshot:
            payload["snapshot"] = snapshot
    unknown_faces.append(payload)


def _should_emit_match_snapshot(user_id):
    if not RETURN_MATCH_SNAPSHOTS:
        return False

    if MATCH_SNAPSHOT_INTERVAL_MS == 0:
        return True

    now_ms = int(time.time() * 1000)
    last_ms = _last_match_snapshot_at.get(user_id, 0)
    if (now_ms - last_ms) < MATCH_SNAPSHOT_INTERVAL_MS:
        return False

    _last_match_snapshot_at[user_id] = now_ms
    return True


def analyze_frame(frame, known_embeddings, known_user_ids):
    global _live_frame_counter

    frame = _resize_if_needed(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    _live_frame_counter += 1

    primary_locations, primary_encodings = _detect_faces_pass(
        rgb,
        PRIMARY_DETECTION_SCALE,
        PRIMARY_UPSAMPLE,
        FRAME_NUM_JITTERS
    )

    face_locations = primary_locations
    face_encodings = primary_encodings

    run_secondary = (_live_frame_counter % SECONDARY_EVERY_N_FRAMES == 0)
    secondary_scale = SECONDARY_DETECTION_SCALE
    secondary_upsample = SECONDARY_UPSAMPLE

    if FAR_SCAN_ON_EMPTY and len(primary_locations) == 0:
        run_secondary = True
        secondary_scale = FAR_DETECTION_SCALE
        secondary_upsample = FAR_UPSAMPLE

    if run_secondary:
        secondary_locations, secondary_encodings = _detect_faces_pass(
            rgb,
            secondary_scale,
            secondary_upsample,
            FRAME_NUM_JITTERS
        )

        face_locations, face_encodings = _merge_detections(
            primary_locations,
            primary_encodings,
            secondary_locations,
            secondary_encodings
        )

    matched = {}
    unknown_faces = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_crop = frame[top:bottom, left:right]

        bbox = {
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "left": int(left)
        }

        if not _valid_face_size(top, right, bottom, left):
            _append_unknown_face(unknown_faces, bbox, face_crop)
            continue

        if len(known_embeddings) == 0:
            _append_unknown_face(unknown_faces, bbox, face_crop)
            continue

        distances = face_recognition.face_distance(known_embeddings, face_encoding)
        if len(distances) == 0:
            _append_unknown_face(unknown_faces, bbox, face_crop)
            continue

        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance > MATCH_THRESHOLD or _is_ambiguous_match(distances):
            _append_unknown_face(unknown_faces, bbox, face_crop)
            continue

        user_id = known_user_ids[best_index]
        confidence = max(0.0, min(1.0, 1.0 - best_distance))

        snapshot = None
        if _should_emit_match_snapshot(user_id):
            snapshot = frame_to_base64(face_crop)

        existing = matched.get(user_id)
        if not existing or confidence > existing["confidence"]:
            matched[user_id] = {
                "userId": user_id,
                "confidence": confidence,
                "snapshot": snapshot,
                "bbox": bbox
            }

    return {
        "matches": list(matched.values()),
        "unknownFaces": unknown_faces
    }


def recognize_faces(video_path):
    cache = load_known_faces()
    known_embeddings = cache["embeddings"]
    known_user_ids = cache["user_ids"]

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

        analysis = analyze_frame(frame, known_embeddings, known_user_ids)
        for match in analysis["matches"]:
            user_id = match["userId"]
            existing = aggregate.get(user_id)
            if not existing or match["confidence"] > existing["confidence"]:
                aggregate[user_id] = match

        frame_idx += 1

    cap.release()
    return list(aggregate.values())


def recognize_single_frame(frame):
    cache = load_known_faces()
    return analyze_frame(frame, cache["embeddings"], cache["user_ids"])


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

    analysis = recognize_single_frame(frame)
    return jsonify(analysis)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
