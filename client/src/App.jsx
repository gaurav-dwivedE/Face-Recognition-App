import React, { useEffect, useRef, useState } from "react";

const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:4000/api";
const POLL_MS = 900;

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

export default function App() {
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState("");
  const [cameraOn, setCameraOn] = useState(false);
  const [liveUserIds, setLiveUserIds] = useState([]);
  const [reg, setReg] = useState({ name: "", studentId: "", image: null });
  const [regStatus, setRegStatus] = useState("Idle");
  const [regError, setRegError] = useState("");

  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const inFlightRef = useRef(false);

  const clearOverlay = () => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  };

  const drawDetections = (matches, unknownList, width, height) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;

    overlay.width = width;
    overlay.height = height;

    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    matches.forEach((match) => {
      if (!match.bbox) return;

      const { left, top, right, bottom } = match.bbox;
      drawBracket(ctx, left, top, right, bottom, "#59c88e");
      drawTarget(ctx, left, top, right, bottom, "#59c88e");

      const label = `${match.user.name} ${(match.confidence * 100).toFixed(0)}%`;
      ctx.font = "13px 'IBM Plex Mono', monospace";
      ctx.fillStyle = "rgba(10, 14, 20, 0.86)";
      const textWidth = ctx.measureText(label).width;
      const labelX = left;
      const labelY = Math.max(18, top - 8);

      ctx.fillRect(labelX - 4, labelY - 15, textWidth + 8, 18);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, labelX, labelY - 2);
    });

    unknownList.forEach((face) => {
      if (!face.bbox) return;

      const { left, top, right, bottom } = face.bbox;
      drawBracket(ctx, left, top, right, bottom, "#ff5d5d");
      drawTarget(ctx, left, top, right, bottom, "#ff5d5d");

      const label = "? Unknown";
      ctx.font = "13px 'IBM Plex Mono', monospace";
      ctx.fillStyle = "rgba(52, 6, 6, 0.86)";
      const textWidth = ctx.measureText(label).width;
      const labelX = left;
      const labelY = Math.max(18, top - 8);
      ctx.fillRect(labelX - 4, labelY - 15, textWidth + 8, 18);
      ctx.fillStyle = "#ffd6d6";
      ctx.fillText(label, labelX, labelY - 2);
    });
  };

  const stopLiveRecognition = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    inFlightRef.current = false;
    clearOverlay();
    setLiveUserIds([]);
    setCameraOn(false);
    setStatus("Stopped");
  };

  const recognizeCurrentFrame = async () => {
    if (!videoRef.current || !captureCanvasRef.current || inFlightRef.current) {
      return;
    }

    const video = videoRef.current;
    if (video.readyState < 2) {
      return;
    }

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 360;

    const captureCanvas = captureCanvasRef.current;
    captureCanvas.width = width;
    captureCanvas.height = height;

    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, width, height);

    inFlightRef.current = true;
    captureCanvas.toBlob(async (blob) => {
      if (!blob) {
        inFlightRef.current = false;
        return;
      }

      try {
        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");

        const response = await fetch(`${apiUrl}/recognize/frame`, {
          method: "POST",
          body: formData
        });

        if (!response.ok) {
          throw new Error("Live recognition failed.");
        }

        const data = await response.json();
        const frameMatches = data.matches || [];
        const frameUnknown = data.unknownFaces || [];

        setResults((prev) => mergePersistentMatches(prev, frameMatches));
        setLiveUserIds(frameMatches.map((item) => item.user.id));
        drawDetections(frameMatches, frameUnknown, width, height);

        setStatus("Running");
        setError("");
      } catch (err) {
        setStatus("Failed");
        setError(err.message || "Live recognition failed.");
      } finally {
        inFlightRef.current = false;
      }
    }, "image/jpeg", 0.85);
  };

  const startLiveRecognition = async () => {
    setError("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera API is not supported in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 15 } },
        audio: false
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraOn(true);
      setStatus("Starting");
      recognizeCurrentFrame();
      timerRef.current = setInterval(recognizeCurrentFrame, POLL_MS);
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
            <span className="subtext">Frame scan every {POLL_MS}ms</span>
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
