"""
Face Recognition API (single endpoint)
----------------------------------------
POST /recognize
  - Upload a captured photo (form-data, key: "file")
  - Compares against a CACHED list of employee face embeddings
    (built once at startup, not re-downloaded/re-processed per request)
  - Returns the matched employee, or "no match"

POST /refresh-cache
  - Re-fetches the employee list from Byposs and rebuilds the embedding
    cache. Call this after adding/updating an employee's face photo.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace
import requests
import numpy as np
import cv2
import io
import traceback

app = FastAPI(title="Face Recognition API")

# Browser preflight (OPTIONS) requests get blocked without this -- the
# Angular dev server (localhost:4200) and anything hitting this through
# the ngrok tunnel both need CORS headers to actually reach /recognize.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend origin(s) before going to production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMPLOYEE_API = "https://app-byposs-backend-linux-gmcvdvf6ecdsg5en.canadacentral-01.azurewebsites.net/api/hrms/v1/PunchControl/employee-face/industry/1"

# Model + threshold. ArcFace is fast and accurate; cosine distance below
# this threshold means "same person". Lower distance = more similar.
MODEL_NAME = "ArcFace"
DISTANCE_THRESHOLD = 0.68  # ArcFace's typical cosine-distance cutoff

# mtcnn: TensorFlow-based face detector, doesn't touch cv2.dnn at all.
# Using this because cv2.dnn is broken on OpenCV 5.0.0 (readNetFromCaffe
# and readNetFromONNX both missing), which killed the ssd/yunet backends.
DETECTOR_BACKEND = "mtcnn"

# In-memory cache: employeeId -> {"embedding": np.ndarray, "employee": dict}
# Built once at startup instead of re-downloading + re-running face
# detection on every employee photo on EVERY /recognize call, which was
# the reason requests were taking 2-3 minutes. Call POST /refresh-cache
# whenever an employee is added or their face photo changes.
EMPLOYEE_CACHE: dict[int, dict] = {}


class MatchResult(BaseModel):
    matched: bool
    employeeId: int | None = None
    employeeName: str | None = None
    employeeCode: str | None = None
    distance: float | None = None
    message: str


class CacheRefreshResult(BaseModel):
    cachedCount: int
    failedCount: int
    message: str


def load_image_from_bytes(image_bytes: bytes):
    """Decode raw image bytes into a numpy array (BGR, OpenCV format)."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def load_image_from_url(url: str):
    """Download an employee's photo and decode it the same way."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return load_image_from_bytes(resp.content)


def cosine_distance(a, b) -> float:
    """Same formula DeepFace uses internally, so DISTANCE_THRESHOLD stays valid."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def build_employee_cache() -> CacheRefreshResult:
    """Fetch the employee list once and precompute+cache each person's
    face embedding, so /recognize never has to re-download or re-run
    face detection on employee photos again."""
    EMPLOYEE_CACHE.clear()

    resp = requests.get(EMPLOYEE_API, timeout=10)
    resp.raise_for_status()
    employees = resp.json().get("data", [])

    failed = 0
    for emp in employees:
        try:
            emp_img = load_image_from_url(emp["faceImage"])
            if emp_img is None:
                failed += 1
                continue

            reps = DeepFace.represent(
                img_path=emp_img,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,  # a bad employee photo shouldn't crash the whole cache build
            )
            if not reps:
                failed += 1
                continue

            EMPLOYEE_CACHE[emp["employeeId"]] = {
                "embedding": reps[0]["embedding"],
                "employee": emp,
            }

        except Exception:
            print(f"Failed to cache embedding for employee {emp.get('employeeName')}:")
            traceback.print_exc()
            failed += 1
            continue

    print(f"Employee cache built: {len(EMPLOYEE_CACHE)} cached, {failed} failed")
    return CacheRefreshResult(
        cachedCount=len(EMPLOYEE_CACHE),
        failedCount=failed,
        message=f"Cached {len(EMPLOYEE_CACHE)} employee(s), {failed} failed.",
    )


@app.on_event("startup")
def startup_build_cache():
    try:
        build_employee_cache()
    except Exception:
        print("Failed to build employee cache at startup:")
        traceback.print_exc()


@app.post("/refresh-cache", response_model=CacheRefreshResult)
async def refresh_cache():
    """Call this after adding a new employee or updating someone's face photo."""
    return build_employee_cache()


@app.post("/recognize", response_model=MatchResult)
async def recognize(file: UploadFile = File(...)):
    captured_bytes = await file.read()
    captured_img = load_image_from_bytes(captured_bytes)

    if captured_img is None:
        return MatchResult(matched=False, message="Could not read the uploaded photo.")

    print(f"Captured image shape: {captured_img.shape}, dtype: {captured_img.dtype}")

    # 0. Reject the photo up front if there's no real face in it.
    # enforce_detection=True here is deliberate and must NEVER be relaxed
    # for the captured photo -- if mtcnn finds no face and we fell back
    # to enforce_detection=False, DeepFace still returns a fallback
    # embedding, which can land deceptively close to a real employee's
    # embedding (false match on a screenshot, blank wall, random object).
    try:
        faces = DeepFace.extract_faces(
            img_path=captured_img,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            anti_spoofing=True,
        )
        if not faces:
            raise ValueError("no face")
    except Exception:
        return MatchResult(matched=False, message="No face detected in the captured photo.")

    if len(faces) > 1:
        return MatchResult(
            matched=False,
            message=f"{len(faces)} faces detected in the photo. Please keep only one person in frame.",
        )

    # anti_spoofing catches non-live faces: printed photos, a photo held up
    # to the camera, a face shown on another screen, etc. DeepFace's
    # MiniFASNet model flags these as is_real=False.
    face_info = faces[0]
    print(f"Anti-spoofing check: is_real={face_info.get('is_real')}, "
          f"antispoof_score={face_info.get('antispoof_score')}, "
          f"confidence={face_info.get('confidence')}")
    if not face_info.get("is_real", True):
        return MatchResult(
            matched=False,
            message="This doesn't look like a live face capture. Please take a real photo in person, not a screen, printed photo, or drawing.",
        )

    if not EMPLOYEE_CACHE:
        return MatchResult(matched=False, message="No employees available to check against.")

    # 1. Get ONE embedding for the captured photo.
    reps = DeepFace.represent(
        img_path=captured_img,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,  # already validated above -- a real single face is present
    )
    captured_embedding = reps[0]["embedding"]

    # 2. Compare against the CACHED employee embeddings -- just math now,
    # no downloads, no re-running face detection on old photos.
    best_match = None
    best_distance = float("inf")

    for entry in EMPLOYEE_CACHE.values():
        distance = cosine_distance(captured_embedding, entry["embedding"])
        if distance < best_distance:
            best_distance = distance
            best_match = entry["employee"]

    # 3. Only accept the match if it's genuinely close -- never force a best guess
    if best_match is not None and best_distance <= DISTANCE_THRESHOLD:
        return MatchResult(
            matched=True,
            employeeId=best_match["employeeId"],
            employeeName=best_match["employeeName"],
            employeeCode=best_match["employeeCode"],
            distance=round(best_distance, 4),
            message="Match found.",
        )

    return MatchResult(
        matched=False,
        distance=round(best_distance, 4) if best_match else None,
        message="No matching employee found.",
    )