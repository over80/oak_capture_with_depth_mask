#!/usr/bin/env

import argparse
import typing
import time
from pathlib import Path

import cv2
import numpy as np
import depthai as dai
import sounddevice as sd

parser = argparse.ArgumentParser()
parser.add_argument(
    "dirpath",
    help="path of directory to create to dump captured images.",
    type=Path,
)

args = parser.parse_args()

print(args)


class IntervalCapture(object):
    SAMPLING_FREQUENCY = 44100

    def __init__(self, dir_path: Path, interval=5.0, pre_beep_num=3):
        self.__dir_path = dir_path
        self.__interval = interval
        self.__pre_beep_num = pre_beep_num
        self.__index = 0
        self.__started = False

        self.__dir_path.mkdir(exist_ok=True)
        (self.__dir_path / "images").mkdir(exist_ok=True)
        (self.__dir_path / "masks").mkdir(exist_ok=True)

        def generate_wave(
            duration_in_sec,
            wave_frequency,
            wave_amplitude=0.3,
            sampling_frequency=self.SAMPLING_FREQUENCY,
        ):
            time_samples = np.arange(0, duration_in_sec, 1 / sampling_frequency)
            wave_samples = (
                np.sin(2 * np.pi * wave_frequency * time_samples) * wave_amplitude
            )
            return wave_samples

        sd.default.samplerate = self.SAMPLING_FREQUENCY
        self.__wave_pre_beep = generate_wave(duration_in_sec=0.2, wave_frequency=440.0)
        self.__wave_beep = generate_wave(duration_in_sec=0.3, wave_frequency=880.0)

    def __trigger_next(self):
        now = time.time()
        self.__next_capture_time = now + self.__interval
        self.__next_pre_beep_time = self.__next_capture_time - 1.0 * self.__pre_beep_num

    def toggle(self):
        if self.__started:
            self.__started = False
        else:
            self.__trigger_next()
            self.__started = True

    def try_pre_beep(self):
        if self.__started:
            now = time.time()
            if (
                self.__next_pre_beep_time is not None
                and now >= self.__next_pre_beep_time
            ):
                sd.play(self.__wave_pre_beep)
                self.__next_pre_beep_time += 1.0
                if abs(self.__next_pre_beep_time - self.__next_capture_time) < 0.5:
                    self.__next_pre_beep_time = None

    def try_capture(
        self,
        frame_rgb: typing.Optional[np.ndarray],
        frame_mask: typing.Optional[np.ndarray],
    ):
        if self.__started:
            now = time.time()
            if now >= self.__next_capture_time:
                if frame_rgb is None or frame_mask is None:
                    # no image to dump.  try again
                    self.__trigger_next()
                    return
                sd.play(self.__wave_beep)
                self.__trigger_next()

                self.__index += 1
                cv2.imwrite(
                    str(self.__dir_path / "images" / f"image_{self.__index}.jpg"),
                    frame_rgb,
                )
                cv2.imwrite(
                    str(self.__dir_path / "masks" / f"image_{self.__index}.jpg.png"),
                    frame_mask,
                )


class BlendRatio(object):
    def __init__(self, window_name):
        self.rgb_weight = 0.3
        cv2.createTrackbar(
            "RGB Weight %",
            window_name,
            int(self.rgb_weight * 100),
            100,
            self.__ui_callback,
        )

    def __ui_callback(self, value):
        self.rgb_weight = value / 100.0

    @property
    def depth_weight(self):
        return 1.0 - self.rgb_weight


class DisparityBorder(object):
    def __init__(self, window_name, max_disparity):
        self.value = 20
        cv2.createTrackbar(
            "mask disparity border",
            window_name,
            self.value,
            int(max_disparity),
            self.__ui_callback,
        )

    def __ui_callback(self, value):
        self.value = value


FPS = 30
MONO_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

# depth ai: generate pipeline
pipeline = dai.Pipeline()
device = dai.Device()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_left = pipeline.create(dai.node.MonoCamera)
cam_right = pipeline.create(dai.node.MonoCamera)
node_stereo_depth = pipeline.create(dai.node.StereoDepth)
out_rgb = pipeline.create(dai.node.XLinkOut)
out_rgb.setStreamName("rgb")
out_disparity = pipeline.create(dai.node.XLinkOut)
out_disparity.setStreamName("disp")

cam_rgb.isp.link(out_rgb.input)
cam_left.out.link(node_stereo_depth.left)
cam_right.out.link(node_stereo_depth.right)
node_stereo_depth.disparity.link(out_disparity.input)

cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(FPS)
try:
    # set fixed focus
    calib_data = device.readCalibration2()
    lens_position = calib_data.getLensPosition(dai.CameraBoardSocket.CAM_A)
    cam_rgb.initialControl.setManualFocus(lens_position)
except:
    raise

cam_left.setResolution(MONO_RESOLUTION)
cam_left.setCamera("left")
cam_left.setFps(FPS)
cam_right.setResolution(MONO_RESOLUTION)
cam_right.setCamera("right")
cam_right.setFps(FPS)

node_stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
node_stereo_depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
node_stereo_depth.setLeftRightCheck(True)  # LR-check is required for depth alignment
node_stereo_depth.setExtendedDisparity(True)
config = node_stereo_depth.initialConfig.get()
# config.algorithmControl.disparityShift = 10  # buggy? with DepthAlign
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.decimationFilter.decimationFactor = 1
node_stereo_depth.initialConfig.set(config)


with device:
    device.startPipeline(pipeline)

    frame_rgb = None
    frame_disparity_colored_1 = None
    frame_disparity_colored_2 = None
    frame_mask = None
    interval_capture = IntervalCapture(args.dirpath)

    max_disparity = node_stereo_depth.initialConfig.getMaxDisparity()

    WINDOW_NAME_RGB = "rgb"
    WINDOW_NAME_DEPTH = "depth"
    WINDOW_NAME_BLENDED = "rgb-depth"
    cv2.namedWindow(WINDOW_NAME_RGB)
    cv2.namedWindow(WINDOW_NAME_DEPTH)
    cv2.namedWindow(WINDOW_NAME_BLENDED)
    cv2.namedWindow("DEBUG")
    blend_ratio = BlendRatio(WINDOW_NAME_BLENDED)
    disparity_border = DisparityBorder(WINDOW_NAME_BLENDED, max_disparity)

    while True:
        latest_packet: dict[str, typing.Optional[typing.Any]] = {}
        latest_packet["rgb"] = None
        latest_packet["disp"] = None

        queue_events = device.getQueueEvents(["rgb", "disp"])
        for queue_name in queue_events:
            packets = device.getOutputQueue(queue_name).tryGetAll()
            if len(packets) > 0:
                latest_packet[queue_name] = packets[-1]

        if latest_packet["rgb"] is not None:
            frame_rgb = latest_packet["rgb"].getCvFrame()
            cv2.imshow(WINDOW_NAME_RGB, frame_rgb)

        if latest_packet["disp"] is not None:
            frame_disparity_raw = latest_packet["disp"].getFrame()
            frame_disparity_grayscale = (
                frame_disparity_raw * 255.0 / max_disparity
            ).astype(np.uint8)
            frame_disparity_colored_1 = np.ascontiguousarray(
                cv2.applyColorMap(frame_disparity_grayscale, cv2.COLORMAP_HOT)
            )
            frame_disparity_colored_2 = np.ascontiguousarray(
                cv2.applyColorMap(frame_disparity_grayscale, cv2.COLORMAP_WINTER)
            )
            cv2.imshow(WINDOW_NAME_DEPTH, frame_disparity_colored_1)

            frame_mask = cv2.cvtColor(
                (frame_disparity_raw > disparity_border.value).astype(np.uint8) * 255,
                cv2.COLOR_GRAY2RGB,
            )
            cv2.imshow("DEBUG", frame_mask)

        # Blend when both received
        if (
            frame_rgb is not None
            and frame_disparity_colored_1 is not None
            and frame_disparity_colored_2 is not None
            and frame_mask is not None
        ):
            blended_nonmasked = np.bitwise_and(
                cv2.addWeighted(
                    frame_rgb,
                    blend_ratio.rgb_weight,
                    frame_disparity_colored_1,
                    blend_ratio.depth_weight,
                    0,
                ),
                frame_mask,
            )
            blended_masked = np.bitwise_and(
                cv2.addWeighted(
                    frame_rgb,
                    blend_ratio.rgb_weight,
                    frame_disparity_colored_2,
                    blend_ratio.depth_weight,
                    0,
                ),
                ~frame_mask,
            )
            blended = np.bitwise_or(blended_masked, blended_nonmasked)

            cv2.imshow(WINDOW_NAME_BLENDED, blended)

            # release images which is only used to display blended image
            frame_disparity_colored_1 = None
            frame_disparity_colored_2 = None

        interval_capture.try_pre_beep()
        interval_capture.try_capture(frame_rgb, frame_mask)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord(" "):
            interval_capture.toggle()
