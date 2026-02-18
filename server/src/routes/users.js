import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import axios from "axios";
import User from "../models/User.js";

const router = express.Router();

const knownDir = path.join(process.cwd(), "..", "shared", "known_faces");
if (!fs.existsSync(knownDir)) {
  fs.mkdirSync(knownDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, knownDir),
  filename: (req, file, cb) => {
    const safeName = `${Date.now()}-${file.originalname}`;
    cb(null, safeName);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Only image files are allowed"));
    }
    cb(null, true);
  }
});

router.post("/", upload.single("image"), async (req, res) => {
  try {
    const { name } = req.body;
    const studentId = req.body.studentId || req.body.employeeId;

    if (!name || !studentId) {
      return res.status(400).json({ error: "Missing name or studentId" });
    }

    if (!req.file?.filename) {
      return res.status(400).json({ error: "Missing image file" });
    }

    const imagePath = path.join("shared", "known_faces", req.file.filename);

    // Store in legacy employeeId column for backward compatibility.
    const user = await User.create({
      name,
      employeeId: studentId,
      imagePath
    });

    const reloadUrl = process.env.PYTHON_RELOAD_SERVICE_URL || "http://localhost:8000/reload-known-faces";
    axios.post(reloadUrl, {}, { timeout: 10000 }).catch(() => {});

    res.status(201).json({
      user: {
        id: String(user._id),
        name: user.name,
        studentId: user.employeeId,
        imagePath: user.imagePath
      }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to register user" });
  }
});

router.get("/", async (req, res) => {
  const users = await User.find({}).lean();
  res.json({
    users: users.map((u) => ({
      id: String(u._id),
      name: u.name,
      studentId: u.studentId || u.employeeId,
      imagePath: u.imagePath,
      metadata: u.metadata
    }))
  });
});

export default router;
