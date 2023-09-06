# image capture script with OAK-D for colmap

## usage

```sh
poetry install
poetry run oak_capture_with_depth_mask.py DIR_PATH_TO_DAMP_IMAGES
```

### controls

- Use the slider at the bottom of the "blended" window to specify the border value of disparity (distance) to be masked
  - Areas marked in blue will be masked, while those marked in red are valid areas.
- Pressing the space key activates the interval timer to capture single image (and corresponding mask) every 5 seconds.
  - Pressing the space key again stops the timer.

## requirements

- PortAudio library(libportaudio2) required for sounddevice python module.
