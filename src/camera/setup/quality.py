import cv2 as cv


def _decode_fourcc(fourcc_value: float) -> str:
    fourcc_int = int(fourcc_value)
    return "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))


def configure_camera_for_quality(
    cap: cv.VideoCapture,
    width: int,
    height: int,
    fps: int,
    prefer_uncompressed: bool,
    manual_exposure: float | None = None,
    manual_gain: float | None = None,
    manual_wb: float | None = None,
    disable_auto_features: bool = True,
    warmup_frames: int = 30
) -> dict:
    """
    Configure an already opened OpenCV VideoCapture for image quality and low latency.

    Sets the pixel format, resolution and frame rate, then the optional manual exposure,
    gain and white balance, disabling the corresponding automatic features first.

    Not every camera, driver or OpenCV backend supports every property, and a ``cap.set(...)``
    that reports success does not guarantee the value was applied. The active settings are
    therefore read back off the device afterwards and returned, so the caller can check what
    it actually got rather than what it asked for.

    Args:
        cap: An already opened VideoCapture.
        width: Desired frame width in pixels.
        height: Desired frame height in pixels.
        fps: Desired frames per second.
        prefer_uncompressed: True tries YUY2 and falls back to YUYV, False uses MJPG.
        manual_exposure: Exposure to set, or None to leave exposure alone.
        manual_gain: Gain to set, or None to leave gain alone.
        manual_wb: White balance to set through ``cv.CAP_PROP_WHITE_BALANCE_BLUE_U``,
            or None to leave white balance alone.
        disable_auto_features: Disable auto exposure, auto white balance and autofocus
            before the manual values are applied.
        warmup_frames: Frames to grab before the settings are read back.

    Returns:
        The settings read back from the device, holding the raw fourcc integer, the same
        value decoded as a four character string, the format actually requested, and the
        image and image-processing properties. Keys: ``fourcc``, ``fourcc_str``,
        ``selected_fourcc_requested``, ``width``, ``height``, ``fps``, ``auto_exposure``,
        ``exposure``, ``gain``, ``auto_wb``, ``wb_blue_u``, ``autofocus``, ``brightness``,
        ``contrast``, ``saturation``, ``sharpness``.

    Raises:
        RuntimeError: ``cap`` is not open. Beyond that, OpenCV or device backend errors
            can still surface from the ``cap.set`` calls.
    """

    if not cap.isOpened():
        raise RuntimeError("Camera is not opened")

    # Preferred pixel format
    preferred_fourccs = ["YUY2", "YUYV"] if prefer_uncompressed else ["MJPG"]
    selected_fourcc = None

    for code in preferred_fourccs:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*code))  # type: ignore[attr-defined]
        actual_code = _decode_fourcc(cap.get(cv.CAP_PROP_FOURCC))
        if actual_code == code:
            selected_fourcc = code
            break

    # Resolution and fps
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)

    # Disable auto features, where the backend implements them at all.
    if disable_auto_features:
        # 0.25 selects manual exposure and 0.75 auto on most backends, but the mapping is
        # backend and camera specific.
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv.CAP_PROP_AUTO_WB, 0)
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

    # Manual values
    if manual_exposure is not None:
        cap.set(cv.CAP_PROP_EXPOSURE, manual_exposure)

    if manual_gain is not None:
        cap.set(cv.CAP_PROP_GAIN, manual_gain)

    if manual_wb is not None:
        cap.set(cv.CAP_PROP_WHITE_BALANCE_BLUE_U, manual_wb)

    # Small buffer for lower latency
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Warm up the camera before the settings are read back.
    for _ in range(warmup_frames):
        cap.grab()
    cap.retrieve()

    actual = {
        "fourcc": int(cap.get(cv.CAP_PROP_FOURCC)),
        "fourcc_str": _decode_fourcc(cap.get(cv.CAP_PROP_FOURCC)),
        "selected_fourcc_requested": selected_fourcc,
        "width": int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv.CAP_PROP_FPS),
        "auto_exposure": cap.get(cv.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv.CAP_PROP_EXPOSURE),
        "gain": cap.get(cv.CAP_PROP_GAIN),
        "auto_wb": cap.get(cv.CAP_PROP_AUTO_WB),
        "wb_blue_u": cap.get(cv.CAP_PROP_WHITE_BALANCE_BLUE_U),
        "autofocus": cap.get(cv.CAP_PROP_AUTOFOCUS),
        "brightness": cap.get(cv.CAP_PROP_BRIGHTNESS),
        "contrast": cap.get(cv.CAP_PROP_CONTRAST),
        "saturation": cap.get(cv.CAP_PROP_SATURATION),
        "sharpness": cap.get(cv.CAP_PROP_SHARPNESS),
    }

    return actual
