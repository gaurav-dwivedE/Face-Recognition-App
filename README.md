# Real-time Face Recognition App

Full-stack app with React, Node/Express, Python (OpenCV + face_recognition), and MongoDB.

## Architecture
- React frontend has registration and live camera recognition.
- Node/Express API accepts frame uploads and fetches user records from MongoDB.
- Python service detects all faces in each frame and compares against registered users.
- MongoDB stores user records and face image paths.

## Setup

### 1) Start MongoDB

```bash
cd /Users/gauravdwivedi/Desktop/face-recognition-app

docker compose up -d
```

### 2) Start the Python service

```bash
cd /Users/gauravdwivedi/Desktop/face-recognition-app/python-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

### 3) Start the Node server

```bash
cd /Users/gauravdwivedi/Desktop/face-recognition-app/server
npm install
cp .env.example .env
npm run dev
```

### 4) Start the React client

```bash
cd /Users/gauravdwivedi/Desktop/face-recognition-app/client
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:5173`.

## Use Flow

1. Register a person from UI with name, employee ID, and clear face image.
2. Click **Start Camera**.
3. App sends live frames to backend every ~1.2s and shows matched users in real time.

## API

- `POST /api/users` (multipart form-data: `name`, `employeeId`, `image`)
- `GET /api/users`
- `POST /api/recognize/frame` (multipart form-data: `frame`)
- `POST /api/recognize` (video upload, still available)

## Notes

- Multiple faces in the same frame are supported.
- Registration auto-refreshes Python embedding cache.
- `face_recognition` depends on `dlib`; install build tools if wheel installation fails.
