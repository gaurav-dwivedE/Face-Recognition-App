import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import axios from "axios";
import FormData from "form-data";
import User from "../models/User.js";

const router = express.Router();

const uploadDir = path.join(process.cwd(), "uploads");
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const safeName = `${Date.now()}-${file.originalname}`;
    cb(null, safeName);
  }
});

const videoUpload = multer({
  storage,
  limits: { fileSize: 200 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith("video/")) {
      return cb(new Error("Only video files are allowed"));
    }
    cb(null, true);
  }
});

const frameUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Only image files are allowed"));
    }
    cb(null, true);
  }
});

const enrichMatches = async (matches) => {
  const userIds = matches.map((m) => m.userId).filter(Boolean);
  if (userIds.length === 0) return [];

  const users = await User.find({ _id: { $in: userIds } }).lean();
  const userMap = new Map(users.map((u) => [String(u._id), u]));

  return matches
    .map((m) => {
      const user = userMap.get(String(m.userId));
      if (!user) return null;
      return {
        user: {
          id: String(user._id),
          name: user.name,
          studentId: user.studentId || user.employeeId,
          imagePath: user.imagePath,
          metadata: user.metadata
        },
        confidence: m.confidence,
        snapshot: m.snapshot,
        bbox: m.bbox || null
      };
    })
    .filter(Boolean);
};

router.post("/", videoUpload.single("video"), async (req, res) => {
  const filePath = req.file?.path;
  if (!filePath) {
    return res.status(400).json({ error: "Missing video file" });
  }

  const pythonUrl = process.env.PYTHON_SERVICE_URL || "http://localhost:8000/recognize";

  try {
    const form = new FormData();
    form.append("video", fs.createReadStream(filePath));

    const pythonRes = await axios.post(pythonUrl, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 300000
    });

    const matches = pythonRes.data?.matches || [];
    const enriched = await enrichMatches(matches);

    res.json({ matches: enriched });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Recognition failed" });
  } finally {
    fs.unlink(filePath, () => {});
  }
});

router.post("/frame", frameUpload.single("frame"), async (req, res) => {
  if (!req.file?.buffer) {
    return res.status(400).json({ error: "Missing frame image" });
  }

  const pythonUrl = process.env.PYTHON_FRAME_SERVICE_URL || "http://localhost:8000/recognize-frame";

  try {
    const form = new FormData();
    form.append("frame", req.file.buffer, {
      filename: req.file.originalname || "frame.jpg",
      contentType: req.file.mimetype
    });

    const pythonRes = await axios.post(pythonUrl, form, {
      headers: form.getHeaders(),
      timeout: 30000
    });

    const matches = pythonRes.data?.matches || [];
    const enriched = await enrichMatches(matches);

    res.json({ matches: enriched });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Live recognition failed" });
  }
});

export default router;
