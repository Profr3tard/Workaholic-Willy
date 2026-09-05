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
    Configure an already opened OpenCV VideoCapture object for higher image quality
    and lower latency.

    The function attempts to set the preferred pixel format (uncompressed YUY2/YUYV
    or compressed MJPG), resolution, frame rate, and several optional camera
    properties such as exposure, gain, and white balance. It also tries to disable
    automatic camera features like auto white balance and autofocus when requested.

    Note:
        Not all cameras, drivers, or OpenCV backends support all properties.
        A successful call to ``cap.set(...)`` does not always guarantee that the
        requested value was actually applied. Therefore, the function reads back
        the active camera settings and returns them for inspection.

    Args:
        cap (cv.VideoCapture):
            An already opened OpenCV VideoCapture object.
        width (int):
            Desired frame width in pixels.
        height (int):
            Desired frame height in pixels.
        fps (int):
            Desired frames per second.
        prefer_uncompressed (bool):
            If True, try to use an uncompressed format (YUY2, fallback YUYV).
            If False, use MJPG.
        manual_exposure (float | None, optional):
            Manual exposure value to set. If None, exposure is not changed manually.
        manual_gain (float | None, optional):
            Manual gain value to set. If None, gain is not changed.
        manual_wb (float | None, optional):
            Manual white balance value to set using
            ``cv.CAP_PROP_WHITE_BALANCE_BLUE_U``. If None, white balance is not
            changed manually.
        disable_auto_features (bool, optional):
            If True, attempts to disable automatic features such as auto exposure,
            auto white balance, and autofocus before applying manual values.
        backend_hint (int | None, optional):
            Optional backend hint for the capture device. Currently unused in this
            function, but kept for API compatibility or future extension.

    Returns:
        dict:
            A dictionary containing the actual camera settings read back from the
            device after configuration. Includes raw and decoded FOURCC as well as
            image and image-processing related properties.

    Keys in returned dictionary:
        - fourcc (int): Raw FOURCC integer value.
        - width (int): Actual frame width.
        - height (int): Actual frame height.
        - fps (float): Actual frame rate.
        - exposure (float): Actual exposure value.
        - gain (float): Actual gain value.
        - wb_blue_u (float): Actual white balance value.
        - brightness (float): Brightness setting.
        - contrast (float): Contrast setting.
        - saturation (float): Saturation setting.
        - sharpness (float): Sharpness setting.
        - fourcc_str (str): FOURCC decoded as a readable 4-character string.

    Raises:
        No exceptions are raised explicitly by this function. However, OpenCV or
        device/backend specific errors may still occur depending on the runtime
        environment.

    Example:
        >>> cap = cv.VideoCapture(0)
        >>> settings = configure_camera_for_quality(
        ...     cap,
        ...     width=1920,
        ...     height=1080,
        ...     fps=30,
        ...     prefer_uncompressed=True,
        ...     manual_exposure=-6,
        ...     manual_gain=0
        ... )
        >>> print(settings["fourcc_str"], settings["width"], settings["height"])
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

    # Disable auto features (backend-dependent!)
    if disable_auto_features:
        # Often: 0.25 = manual, 0.75 = auto (depends on backend/camera)
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

    # warmup for camera:
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
