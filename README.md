# lowlight-occlusion

Simulate **low-light** and **occlusion** degradations for videos. Useful for generating stress-test data for computer vision models.

## What it does

- Reads videos from a source folder (recursively)
- Writes two new videos per input:
	- Low-light version (brightness scaled in HSV)
	- Occlusion version (random gray rectangles that move every N frames)

## Requirements

- Python 3.8+
- Packages:
	- opencv-python
	- numpy

Install:

```bash
pip install opencv-python numpy
```

## Quick start

1. Put your videos under the folder configured by `SOURCE_FOLDER` in `video_process.py`.
2. Run:

```bash
python video_process.py
```

If the input folder does not exist, the script will create a small dummy test video and exit. Re-run after that.

## Output

Given:

- Input root: `splited_video_all629/`

It generates:

- `splited_video_all629_LowLight/` (same subfolder structure)
- `splited_video_all629_Occlusion/` (same subfolder structure)

## Configuration

All settings are in the `CONFIG` dict inside `video_process.py`:

- `SOURCE_FOLDER`: input root folder
- `VIDEO_EXTENSIONS`: file extensions to include
- `NUM_PROCESSES`: number of worker processes (`None` = all CPU cores)
- `LOW_LIGHT_FACTOR`: brightness scale (e.g., `0.3` is darker)
- Occlusion:
	- `OCCLUSION_CHANGE_INTERVAL`: move blocks every N frames
	- `OCCLUSION_MAX_COUNT`: max blocks per frame
	- `OCCLUSION_MIN_SIZE` / `OCCLUSION_MAX_SIZE`: block size range (pixels)
	- `OCCLUSION_COLOR`: BGR color (default dark gray)
