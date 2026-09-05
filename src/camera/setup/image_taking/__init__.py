"""The frame dataclasses every rig returns, and the streamers that produce them."""

from __future__ import annotations

from .frames import AnyFrame, RGBDFrame, StereoFrame
from .rgbd import OpenCvRGBDStreamer, RealSenseRGBDStreamer, RGBDStreamerProtocol
from .single import SingleDeviceStreamer
from .webcam import WebcamPairStreamer

__all__ = [
	"AnyFrame",
	"OpenCvRGBDStreamer",
	"RGBDFrame",
	"RGBDStreamerProtocol",
	"RealSenseRGBDStreamer",
	"SingleDeviceStreamer",
	"StereoFrame",
	"WebcamPairStreamer",
]
