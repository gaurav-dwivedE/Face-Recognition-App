import mongoose from "mongoose";

const UserSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
    employeeId: { type: String, required: true, unique: true },
    imagePath: { type: String, required: true },
    metadata: { type: Object, default: {} }
  },
  { timestamps: true }
);

export default mongoose.model("User", UserSchema);
