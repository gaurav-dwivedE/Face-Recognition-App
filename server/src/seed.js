import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import mongoose from "mongoose";
import User from "./models/User.js";

dotenv.config();

const mongoUri = process.env.MONGO_URI || "mongodb://localhost:27017/face_recognition";
const dataPath = process.env.SEED_JSON || path.join(process.cwd(), "data", "users.json");

const run = async () => {
  const raw = fs.readFileSync(dataPath, "utf-8");
  const users = JSON.parse(raw);

  await mongoose.connect(mongoUri);

  await User.deleteMany({});
  await User.insertMany(users);

  console.log(`Seeded ${users.length} users`);
  await mongoose.disconnect();
};

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
