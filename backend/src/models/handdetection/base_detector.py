import asyncio
import threading

# mediapipe is an OPTIONAL dependency (hand/gesture detection is NOT part of the grasp pipeline).
# Install it via requirements/voice.txt. Imported in a guard so this module imports cleanly without it;
# constructing a detector without mediapipe raises a clear, actionable error (see __init__).
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python.vision import (
        GestureRecognizer,
        GestureRecognizerOptions,
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    _MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover - only on installs without the optional voice extra
    mp = None  # type: ignore[assignment]
    python = None  # type: ignore[assignment]
    GestureRecognizer = GestureRecognizerOptions = None  # type: ignore[assignment,misc]
    HandLandmarker = HandLandmarkerOptions = RunningMode = None  # type: ignore[assignment,misc]
    _MEDIAPIPE_AVAILABLE = False

from backend.config.schema.models.models_schema import GestureDetectConfig, HandDetectConfig
from backend.src.utility.vision import bgr_to_rgb


class BaseHandDetector:
    """Base class for MediaPipe LIVE_STREAM hand + gesture detection.

    The MediaPipe LIVE_STREAM running mode invokes ``result_callback``
    from MediaPipe's internal worker thread while the main pipeline
    thread reads :attr:`hands`, :attr:`frame_landmarks`, :attr:`gesture`
    via :meth:`snapshot` (or the legacy attribute reads). All shared
    state is therefore guarded by :attr:`_lock` (a ``threading.Lock``);
    use :meth:`asnapshot` for an async-safe variant.
    """

    def __init__(self, hand_config: HandDetectConfig, gesture_config: GestureDetectConfig):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is required for hand/gesture detection but is not installed. "
                "Install the optional voice/gesture extra: pip install -r requirements/voice.txt"
            )
        # ---------------------------------------------------------
        # Shared state (populated by both async callbacks).
        # All reads/writes guarded by self._lock.
        # ---------------------------------------------------------
        self._lock = threading.Lock()
        self.hands: list[list[tuple[int, int]]] = []
        self.frame_landmarks: list[list[tuple[int, int]]] = []
        self.gesture = None             # Latest recognised gesture label
        self.timestamp = 0              # Shared monotonic timestamp (ms)

        # ---------------------------------------------------------
        # HandLandmarker callback (called on a MediaPipe worker thread)
        # ---------------------------------------------------------
        def _hands_callback(result, img, ts):
            new_hands: list = []
            new_lms: list = []

            if result and result.hand_landmarks:
                rgb = img.numpy_view()
                h, w = rgb.shape[:2]
                for hand in result.hand_landmarks:
                    lm = [(int(p.x * w), int(p.y * h)) for p in hand]
                    new_hands.append(lm)
                    new_lms.append(lm)

            with self._lock:
                self.hands = new_hands
                self.frame_landmarks = new_lms

        # ---------------------------------------------------------
        # GestureRecognizer callback (also called on a worker thread)
        # ---------------------------------------------------------
        def _gesture_callback(result, img, ts):
            new_gesture = None
            if result and result.gestures:
                new_gesture = result.gestures[0][0].category_name
            with self._lock:
                self.gesture = new_gesture

        # ---------------------------------------------------------
        # MediaPipe HandLandmarker
        # ---------------------------------------------------------
        hand_opts = HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_config.model_path),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=_hands_callback,
            num_hands=hand_config.max_hands,
            min_hand_detection_confidence=hand_config.threshold,
        )
        self.detector_hands = HandLandmarker.create_from_options(hand_opts)

        gesture_opts = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=gesture_config.model_path),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=_gesture_callback,
        )
        self.detector_gesture = GestureRecognizer.create_from_options(gesture_opts)

    # ---------------------------------------------------------
    # Shared detect(): fires both MediaPipe models asynchronously
    # ---------------------------------------------------------
    def detect(self, frame_bgr):
        rgb = bgr_to_rgb(frame_bgr)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._lock:
            ts = self.timestamp
            self.timestamp = ts + 1

        # Fire both MediaPipe models asynchronously on the same frame.
        self.detector_hands.detect_async(img, ts)
        self.detector_gesture.recognize_async(img, ts)

        return frame_bgr

    # ---------------------------------------------------------
    # Thread-safe accessors
    # ---------------------------------------------------------
    def snapshot(self):
        """Return a consistent ``(hands, frame_landmarks, gesture)`` tuple.

        Synchronous; safe to call from any thread. The returned lists
        are shallow copies, so callers can iterate without worrying
        about MediaPipe callbacks mutating them mid-iteration.
        """
        with self._lock:
            return (
                list(self.hands),
                list(self.frame_landmarks),
                self.gesture,
            )

    async def asnapshot(self):
        """Async wrapper around :meth:`snapshot` (offloaded to a thread).

        The lock is short-held, so this is essentially equivalent to
        :meth:`snapshot` with the convenience of being awaitable from
        an event loop without blocking it.
        """
        return await asyncio.to_thread(self.snapshot)
