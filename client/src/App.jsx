import React, { useEffect, useRef, useState } from "react";

const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:4000/api";
const TARGET_POLL_MS = 90;
const CAPTURE_MAX_WIDTH = 1600;
const JPEG_QUALITY = 0.8;
const KNOWN_TRACK_TTL_MS = 900;
const UNKNOWN_TRACK_TTL_MS = 520;
const TRACK_SMOOTH_ALPHA = 0.52;
const TRACK_PREDICT_MS = 120;
const LOCAL_DETECT_MS = 34;

const mergePersistentMatches = (previous, incoming) => {
  const now = Date.now();
  const map = new Map(previous.map((item) => [item.user.id, item]));

  incoming.forEach((item) => {
    const key = item.user.id;
    const existing = map.get(key);

    if (!existing) {
      map.set(key, { ...item, lastSeen: now });
      return;
    }

    map.set(key, {
      ...existing,
      ...item,
      confidence: Math.max(existing.confidence || 0, item.confidence || 0),
      snapshot: item.snapshot || existing.snapshot,
      lastSeen: now
    });
  });

  return Array.from(map.values()).sort((a, b) => (b.lastSeen || 0) - (a.lastSeen || 0));
};

const drawBracket = (ctx, left, top, right, bottom, color) => {
  const width = right - left;
  const height = bottom - top;
  const len = Math.max(12, Math.min(width, height) * 0.22);

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;

  ctx.beginPath();
  ctx.moveTo(left, top + len);
  ctx.lineTo(left, top);
  ctx.lineTo(left + len, top);
  ctx.moveTo(right - len, top);
  ctx.lineTo(right, top);
  ctx.lineTo(right, top + len);
  ctx.moveTo(left, bottom - len);
  ctx.lineTo(left, bottom);
  ctx.lineTo(left + len, bottom);
  ctx.moveTo(right - len, bottom);
  ctx.lineTo(right, bottom);
  ctx.lineTo(right, bottom - len);
  ctx.stroke();
};

const drawTarget = (ctx, left, top, right, bottom, color) => {
  const centerX = Math.round((left + right) / 2);
  const centerY = Math.round((top + bottom) / 2);
  const radius = Math.max(10, Math.min(right - left, bottom - top) * 0.18);

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.moveTo(centerX - radius - 6, centerY);
  ctx.lineTo(centerX - radius + 6, centerY);
  ctx.moveTo(centerX + radius - 6, centerY);
  ctx.lineTo(centerX + radius + 6, centerY);
  ctx.moveTo(centerX, centerY - radius - 6);
  ctx.lineTo(centerX, centerY - radius + 6);
  ctx.moveTo(centerX, centerY + radius - 6);
  ctx.lineTo(centerX, centerY + radius + 6);
  ctx.stroke();
};

const formatLastSeen = (timestamp) => {
  if (!timestamp) return "-";
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
};

const scaleDetections = (items, scaleX, scaleY) =>
  items
    .map((item) => {
      if (!item.bbox) return item;

      return {
        ...item,
        bbox: {
          left: item.bbox.left * scaleX,
          top: item.bbox.top * scaleY,
          right: item.bbox.right * scaleX,
          bottom: item.bbox.bottom * scaleY
        }
      };
    })
    .filter((item) => item.bbox);

const clampBbox = (bbox, width, height) => {
  const left = Math.max(0, Math.min(width - 2, bbox.left));
  const top = Math.max(0, Math.min(height - 2, bbox.top));
  const right = Math.max(left + 2, Math.min(width, bbox.right));
  const bottom = Math.max(top + 2, Math.min(height, bbox.bottom));

  return { left, top, right, bottom };
};

const bboxIoU = (a, b) => {
  const interLeft = Math.max(a.left, b.left);
  const interTop = Math.max(a.top, b.top);
  const interRight = Math.min(a.right, b.right);
  const interBottom = Math.min(a.bottom, b.bottom);

  const iw = Math.max(0, interRight - interLeft);
  const ih = Math.max(0, interBottom - interTop);
  const intersection = iw * ih;

  const areaA = Math.max(0, a.right - a.left) * Math.max(0, a.bottom - a.top);
  const areaB = Math.max(0, b.right - b.left) * Math.max(0, b.bottom - b.top);
  const union = areaA + areaB - intersection;

  if (union <= 0) return 0;
  return intersection / union;
};

const lerp = (a, b, alpha) => a + (b - a) * alpha;

const smoothBbox = (current, target, alpha) => ({
  left: lerp(current.left, target.left, alpha),
  top: lerp(current.top, target.top, alpha),
  right: lerp(current.right, target.right, alpha),
  bottom: lerp(current.bottom, target.bottom, alpha)
});

const drawTrackLabel = (ctx, track) => {
  const label = track.label;
  ctx.font = "13px 'IBM Plex Mono', monospace";
  ctx.fillStyle = track.labelBg;
  const textWidth = ctx.measureText(label).width;
  const labelX = track.current.left;
  const labelY = Math.max(18, track.current.top - 8);

  ctx.fillRect(labelX - 4, labelY - 15, textWidth + 8, 18);
  ctx.fillStyle = track.labelColor;
  ctx.fillText(label, labelX, labelY - 2);
};

const makeUnknownPayload = (bbox) => ({
  bbox,
  label: "? Unknown",
  color: "#ff5d5d",
  labelBg: "rgba(52, 6, 6, 0.86)",
  labelColor: "#ffd6d6"
});

const localFaceBBox = (detection) => {
  const box = detection?.boundingBox;
  if (!box) return null;

  const left = Number(box.x ?? box.left ?? 0);
  const top = Number(box.y ?? box.top ?? 0);
  const width = Number(box.width ?? 0);
  const height = Number(box.height ?? 0);

  if (!(width > 0 && height > 0)) return null;

  return {
    left,
    top,
    right: left + width,
    bottom: top + height
  };
};

export default function App() {
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState("");
  const [cameraOn, setCameraOn] = useState(false);
  const [liveUserIds, setLiveUserIds] = useState([]);
  const [reg, setReg] = useState({ name: "", studentId: "", image: null });
  const [regStatus, setRegStatus] = useState("Idle");
  const [regError, setRegError] = useState("");
  const [trackerMode, setTrackerMode] = useState("Server");

  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const rafRef = useRef(null);
  const localDetectTimerRef = useRef(null);
  const localDetectInFlightRef = useRef(false);
  const inFlightRef = useRef(false);
  const isRunningRef = useRef(false);
  const trackMapRef = useRef(new Map());
  const unknownTrackIdRef = useRef(1);
  const faceDetectorRef = useRef(null);

  const clearOverlay = () => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  };

  const stopOverlayLoop = () => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  };

  const stopLocalDetectLoop = () => {
    if (localDetectTimerRef.current) {
      clearTimeout(localDetectTimerRef.current);
      localDetectTimerRef.current = null;
    }
    localDetectInFlightRef.current = false;
  };

  const updateTrackSignal = (track, bbox, now) => {
    const dt = Math.max(1, now - (track.lastSignalAt || now));

    track.velocity = {
      left: (bbox.left - track.target.left) / dt,
      top: (bbox.top - track.target.top) / dt,
      right: (bbox.right - track.target.right) / dt,
      bottom: (bbox.bottom - track.target.bottom) / dt
    };
    track.target = { ...bbox };
    track.lastSignalAt = now;
    track.lastSeen = now;
  };

  const upsertTrack = (key, payload, kind) => {
    const now = Date.now();
    const existing = trackMapRef.current.get(key);

    if (!existing) {
      const initial = { ...payload.bbox };
      trackMapRef.current.set(key, {
        key,
        kind,
        color: payload.color,
        labelBg: payload.labelBg,
        labelColor: payload.labelColor,
        label: payload.label,
        current: initial,
        target: initial,
        velocity: { left: 0, top: 0, right: 0, bottom: 0 },
        lastSeen: now,
        lastSignalAt: now
      });
      return;
    }

    existing.label = payload.label;
    existing.color = payload.color;
    existing.labelBg = payload.labelBg;
    existing.labelColor = payload.labelColor;
    updateTrackSignal(existing, payload.bbox, now);
  };

  const touchTrack = (track, bbox) => {
    const now = Date.now();
    updateTrackSignal(track, bbox, now);
  };

  const drawTracks = () => {
    const video = videoRef.current;
    const overlay = overlayCanvasRef.current;
    if (!video || !overlay) return;

    const width = video.videoWidth || 0;
    const height = video.videoHeight || 0;
    if (!width || !height) return;

    if (overlay.width !== width || overlay.height !== height) {
      overlay.width = width;
      overlay.height = height;
    }

    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const now = Date.now();
    const toDelete = [];

    trackMapRef.current.forEach((track, key) => {
      const ttl = track.kind === "known" ? KNOWN_TRACK_TTL_MS : UNKNOWN_TRACK_TTL_MS;
      if (now - track.lastSeen > ttl) {
        toDelete.push(key);
        return;
      }

      const predictMs = Math.min(TRACK_PREDICT_MS, now - (track.lastSignalAt || now));
      const predicted = {
        left: track.target.left + track.velocity.left * predictMs,
        top: track.target.top + track.velocity.top * predictMs,
        right: track.target.right + track.velocity.right * predictMs,
        bottom: track.target.bottom + track.velocity.bottom * predictMs
      };

      const boundedPredicted = clampBbox(predicted, width, height);
      track.current = smoothBbox(track.current, boundedPredicted, TRACK_SMOOTH_ALPHA);
      track.current = clampBbox(track.current, width, height);

      drawBracket(ctx, track.current.left, track.current.top, track.current.right, track.current.bottom, track.color);
      drawTarget(ctx, track.current.left, track.current.top, track.current.right, track.current.bottom, track.color);
      drawTrackLabel(ctx, track);
    });

    toDelete.forEach((key) => trackMapRef.current.delete(key));
  };

  const overlayLoop = () => {
    if (!isRunningRef.current) return;
    drawTracks();
    rafRef.current = requestAnimationFrame(overlayLoop);
  };

  const startOverlayLoop = () => {
    stopOverlayLoop();
    rafRef.current = requestAnimationFrame(overlayLoop);
  };

  const updateTracksFromRecognition = (matches, unknownFaces) => {
    const unknownCandidates = [];
    trackMapRef.current.forEach((track) => {
      if (track.kind === "unknown") {
        unknownCandidates.push(track);
      }
    });

    const usedUnknownTrackKeys = new Set();

    matches.forEach((match) => {
      if (!match.bbox) return;
      const key = `known-${match.user.id}`;

      upsertTrack(
        key,
        {
          bbox: match.bbox,
          label: `${match.user.name} ${(match.confidence * 100).toFixed(0)}%`,
          color: "#59c88e",
          labelBg: "rgba(10, 14, 20, 0.86)",
          labelColor: "#ffffff"
        },
        "known"
      );
    });

    unknownFaces.forEach((face) => {
      if (!face.bbox) return;

      let matchedTrack = null;
      let bestScore = 0.1;

      for (const candidate of unknownCandidates) {
        if (usedUnknownTrackKeys.has(candidate.key)) continue;
        const score = bboxIoU(candidate.target, face.bbox);
        if (score > bestScore) {
          bestScore = score;
          matchedTrack = candidate;
        }
      }

      const key = matchedTrack?.key || `unknown-${unknownTrackIdRef.current++}`;
      usedUnknownTrackKeys.add(key);

      upsertTrack(key, makeUnknownPayload(face.bbox), "unknown");
    });
  };

  const updateTracksFromLocalDetection = (detections, width, height) => {
    if (detections.length === 0) return;

    const usedTrackKeys = new Set();

    detections.forEach((det) => {
      const rawBbox = localFaceBBox(det);
      if (!rawBbox) return;

      const bbox = clampBbox(rawBbox, width, height);
      let bestTrack = null;
      let bestScore = 0;

      trackMapRef.current.forEach((track) => {
        if (usedTrackKeys.has(track.key)) return;

        const threshold = track.kind === "known" ? 0.08 : 0.12;
        const score = bboxIoU(track.target, bbox);
        if (score > threshold && score > bestScore) {
          bestScore = score;
          bestTrack = track;
        }
      });

      if (bestTrack) {
        touchTrack(bestTrack, bbox);
        usedTrackKeys.add(bestTrack.key);
        return;
      }

      const key = `unknown-${unknownTrackIdRef.current++}`;
      upsertTrack(key, makeUnknownPayload(bbox), "unknown");
      usedTrackKeys.add(key);
    });
  };

  const runLocalDetectLoop = async () => {
    if (!isRunningRef.current) return;

    const detector = faceDetectorRef.current;
    const video = videoRef.current;

    if (
      detector &&
      video &&
      video.readyState >= 2 &&
      !localDetectInFlightRef.current &&
      (video.videoWidth || 0) > 0 &&
      (video.videoHeight || 0) > 0
    ) {
      localDetectInFlightRef.current = true;

      try {
        const detections = await detector.detect(video);
        if (isRunningRef.current) {
          updateTracksFromLocalDetection(detections || [], video.videoWidth, video.videoHeight);
        }
      } catch {
        // Ignore detector errors and keep server recognition running.
      } finally {
        localDetectInFlightRef.current = false;
      }
    }

    if (!isRunningRef.current) return;
    localDetectTimerRef.current = setTimeout(() => {
      runLocalDetectLoop();
    }, LOCAL_DETECT_MS);
  };

  const stopLiveRecognition = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    isRunningRef.current = false;
    inFlightRef.current = false;
    stopOverlayLoop();
    stopLocalDetectLoop();
    faceDetectorRef.current = null;
    trackMapRef.current.clear();
    clearOverlay();
    setLiveUserIds([]);
    setCameraOn(false);
    setStatus("Stopped");
    setTrackerMode("Server");
  };

  const recognizeCurrentFrame = async () => {
    if (!videoRef.current || !captureCanvasRef.current || inFlightRef.current || !isRunningRef.current) {
      return;
    }

    const video = videoRef.current;
    if (video.readyState < 2) {
      return;
    }

    const sourceWidth = video.videoWidth || 640;
    const sourceHeight = video.videoHeight || 360;
    const sendScale = sourceWidth > CAPTURE_MAX_WIDTH ? CAPTURE_MAX_WIDTH / sourceWidth : 1;
    const sendWidth = Math.max(320, Math.round(sourceWidth * sendScale));
    const sendHeight = Math.max(180, Math.round(sourceHeight * sendScale));

    const captureCanvas = captureCanvasRef.current;
    captureCanvas.width = sendWidth;
    captureCanvas.height = sendHeight;

    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, sendWidth, sendHeight);

    inFlightRef.current = true;

    try {
      const blob = await new Promise((resolve) => {
        captureCanvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY);
      });

      if (!blob) {
        return;
      }

      const formData = new FormData();
      formData.append("frame", blob, "frame.jpg");

      const response = await fetch(`${apiUrl}/recognize/frame`, {
        method: "POST",
        body: formData,
        cache: "no-store"
      });

      if (!response.ok) {
        throw new Error("Live recognition failed.");
      }

      const data = await response.json();
      const frameMatches = data.matches || [];
      const frameUnknown = data.unknownFaces || [];

      const scaleX = sourceWidth / sendWidth;
      const scaleY = sourceHeight / sendHeight;
      const scaledMatches = scaleDetections(frameMatches, scaleX, scaleY);
      const scaledUnknown = scaleDetections(frameUnknown, scaleX, scaleY);

      updateTracksFromRecognition(scaledMatches, scaledUnknown);
      setResults((prev) => mergePersistentMatches(prev, scaledMatches));
      setLiveUserIds(scaledMatches.map((item) => item.user.id));

      setStatus("Running");
      setError("");
    } catch (err) {
      setStatus("Failed");
      setError(err.message || "Live recognition failed.");
    } finally {
      inFlightRef.current = false;
    }
  };

  const runRecognitionLoop = async () => {
    if (!isRunningRef.current) return;

    const startedAt = performance.now();
    await recognizeCurrentFrame();

    if (!isRunningRef.current) return;

    const elapsed = performance.now() - startedAt;
    const delay = Math.max(0, TARGET_POLL_MS - elapsed);
    timerRef.current = setTimeout(() => {
      runRecognitionLoop();
    }, delay);
  };

  const startLiveRecognition = async () => {
    setError("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera API is not supported in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 60, min: 24 }
        },
        audio: false
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      faceDetectorRef.current = null;
      if ("FaceDetector" in window) {
        try {
          faceDetectorRef.current = new window.FaceDetector({
            fastMode: true,
            maxDetectedFaces: 20
          });
        } catch {
          faceDetectorRef.current = null;
        }
      }

      trackMapRef.current.clear();
      unknownTrackIdRef.current = 1;
      isRunningRef.current = true;
      setCameraOn(true);
      setStatus("Starting");
      setTrackerMode(faceDetectorRef.current ? "Server + Local" : "Server");

      startOverlayLoop();
      if (faceDetectorRef.current) {
        runLocalDetectLoop();
      }
      runRecognitionLoop();
    } catch {
      setError("Unable to access camera. Allow camera permission and retry.");
      setStatus("Failed");
    }
  };

  const handleRegister = async (event) => {
    event.preventDefault();
    setRegError("");

    if (!reg.name || !reg.studentId || !reg.image) {
      setRegError("Name, student ID, and image are required.");
      return;
    }

    setRegStatus("Uploading");

    try {
      const formData = new FormData();
      formData.append("name", reg.name);
      formData.append("studentId", reg.studentId);
      formData.append("image", reg.image);

      const response = await fetch(`${apiUrl}/users`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error("Registration failed.");
      }

      setRegStatus("Registered");
      setReg({ name: "", studentId: "", image: null });
    } catch (err) {
      setRegStatus("Failed");
      setRegError(err.message || "Registration failed.");
    }
  };

  useEffect(() => () => stopLiveRecognition(), []);

  return (
    <div className="app">
      <section className="card register-card">
        <div className="section-head">
          <h2>Registration</h2>
          <span className="subtext">Add a person with one clear face image</span>
        </div>
        <form className="register-form" onSubmit={handleRegister}>
          <label className="field">
            <span>Name</span>
            <input
              type="text"
              value={reg.name}
              onChange={(e) => setReg({ ...reg, name: e.target.value })}
              placeholder="Full name"
            />
          </label>
          <label className="field">
            <span>Student ID</span>
            <input
              type="text"
              value={reg.studentId}
              onChange={(e) => setReg({ ...reg, studentId: e.target.value })}
              placeholder="STU-123"
            />
          </label>
          <label className="file-input">
            <span>Face image</span>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setReg({ ...reg, image: e.target.files?.[0] || null })}
            />
          </label>
          <button type="submit">Register</button>
          <div className="status">Registration: {regStatus}</div>
          {regError && <div className="error">{regError}</div>}
        </form>
      </section>

      <section className="live-layout">
        <section className="card">
          <div className="section-head">
            <h2>Live Camera</h2>
            <span className="subtext">Smooth tracking mode: {trackerMode}</span>
          </div>
          <div className="camera-box">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={overlayCanvasRef} className="overlay-canvas" />
            <canvas ref={captureCanvasRef} className="hidden-canvas" />
          </div>
          <div className="controls">
            <button onClick={startLiveRecognition} disabled={cameraOn}>
              Start
            </button>
            <button className="secondary" onClick={stopLiveRecognition} disabled={!cameraOn}>
              Stop
            </button>
            <span className="status-pill">Status: {status}</span>
          </div>
          {error && <div className="error">{error}</div>}
        </section>

        <section className="card table-card">
          <div className="section-head">
            <h2>Detected Persons</h2>
            <span className="subtext">Entries remain after person leaves frame</span>
          </div>
          {results.length === 0 ? (
            <div className="empty">No detections yet.</div>
          ) : (
            <div className="table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Photo</th>
                    <th>Name</th>
                    <th>Student ID</th>
                    <th>Confidence</th>
                    <th>State</th>
                    <th>Last Seen</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((item) => (
                    <tr key={item.user.id}>
                      <td>
                        {item.snapshot ? (
                          <img
                            className="table-face"
                            src={`data:image/jpeg;base64,${item.snapshot}`}
                            alt={item.user.name}
                          />
                        ) : (
                          <div className="table-face table-face-empty">-</div>
                        )}
                      </td>
                      <td>{item.user.name}</td>
                      <td>{item.user.studentId}</td>
                      <td>{(item.confidence * 100).toFixed(2)}%</td>
                      <td>
                        {liveUserIds.includes(item.user.id) ? (
                          <span className="live-pill">Live</span>
                        ) : (
                          <span className="seen-pill">Seen</span>
                        )}
                      </td>
                      <td>{formatLastSeen(item.lastSeen)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </section>
    </div>
  );
}
