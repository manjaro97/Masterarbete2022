# Masterarbete2022 — Video-based Aim-Point Analysis and Visualization
https://urn.kb.se/resolve?urn=urn:nbn:se:miun:diva-48041

## Summary
This project processes a video of a target and an aim-point to track the target, compute aim-point statistics (offsets, distance, velocity, acceleration, precision, accuracy) and produce visualizations (overlaid video frames, time-series graphs, polar plots and a composite "final" layout). It uses OpenCV for image/video I/O and drawing, NumPy/math for numeric operations, and Matplotlib (FigureCanvasAgg) to render plots which are converted into OpenCV images and overlaid on frames.

---

## Features / What this code does
- Detects a target in each frame using OpenCV template matching.
- Tracks a given aim-point relative to the detected target center and stores a history of offsets.
- Computes:
  - Euclidean distance between aim-point and target center.
  - Velocity and acceleration (finite differences).
  - Running (cumulative) mean offset in X and Y.
  - Precision and accuracy percentages over a recent window of frames.
- Renders visual outputs:
  - Bounding box and aim-point rendered directly on frames.
  - Time-series plots (offset, mean offset, precision, accuracy, Euclidean distance, velocity, acceleration) drawn by Matplotlib and overlaid on frames.
  - Polar plot showing recent aim-point path on a polar axis.
  - Composed layouts for editing and final output videos.
- Writes multiple output videos:
  - editedVid — edited video with layout & overlays
  - graphVid — large canvas with multiple plots
  - finalVid — final composite layout
  - polarPlot — polar visualization video
- Shows a live preview window while processing.

---

## Quick start / Usage
1. Install dependencies (examples; see requirements.txt in repo):
   - numpy
   - opencv-python (cv2)
   - matplotlib
2. Place input assets in `input/`:
   - target image(s) like `target4.jpg`
   - corresponding video(s) like `kct4.mp4`
3. Run the main script:
   - python main.py
4. Output videos are written to `output/` (created by the script if not present). Temporary per-frame JPEGs are written to `frames/`.

---

## High-level flow
1. Open the chosen video file with OpenCV VideoCapture.
2. For each frame:
   - Save frame as JPEG and re-load (the code currently writes and reads from disk).
   - Find target location with `cv.matchTemplate(..., method=cv.TM_CCOEFF_NORMED)`.
   - Compute target centroid and the aim-point offset (relative vector).
   - Append offset to history arrays.
   - Compute Euclidean distance, velocity (difference), acceleration (difference of velocity).
   - Compute cumulative mean offset.
   - Compute precision and accuracy over the last N frames (N=30 by default).
   - Render Matplotlib plots (line/polar) into RGBA buffers and convert to OpenCV BGR.
   - Overlay plots and draw graphics/text directly on frames.
   - Write frames into the configured VideoWriters.
3. Clean up writers and windows.

---

## Important scripts / files
- `main.py` — Main pipeline that processes the video and orchestrates analysis and rendering.
- `ias.py` — (Present in repo — auxiliary module; not covered in detail in this README).
- `polarPlot.py` — Small helper (present in repo) for polar plotting (the main code renders polar plots in `main.py`).
- `requirements.txt` — Lists Python packages needed (check file for exact versions).
- `input/` — Folder for target images and videos used by the script.
- `frames/` — Temporary per-frame images (created/used by `main.py`).
- `output/` — Destination for generated videos.

---

## Key functions in main.py (what they do)
- drawTargetAndAim(frame, aimpointPos, targetTopLeftPos, targetBottomRightPos)
  - Draws bounding rectangle of detected target and a small circle at the aim-point. Uses green if aim-point is inside the box; red otherwise.

- drawPlot(frame, dataArray, plotPos, plotType)
  - Creates a simple Matplotlib Figure, plots `dataArray`, formats axes/titles depending on `plotType`, renders to an RGBA buffer and overlays it on `frame` starting at `plotPos`.
  - plotType influences axis labels, legends, scaling and y-limits. The x-axis formatter converts frame index → seconds via division by 30.

- drawPolarPlot(frame, aimpointOffsetArray, plotPos)
  - Converts offset history into polar coordinates (r, theta), splits the data into segments to color-code older vs recent points, draws with Matplotlib polar axes, renders and overlays into `frame`.

- drawMeanOffset(aimpointOffsetMeanArray, frame, pos)
  - Writes the cumulative mean X/Y offset to the frame as text.

- drawTime(time, frame, pos)
  - Writes a time stamp (seconds) on the frame.

- drawLayout(frame, centroid) / drawFinalLayout(...)
  - Draw a background and visual layout for composed frames (background color, borders, slots for plots). `drawFinalLayout` prepares the final composition with edges and designated plot positions.

- zoom_at(frame, zoom=2, angle=0, coord=None)
  - Uses `cv.getRotationMatrix2D` and `cv.warpAffine` to zoom and optionally rotate around a center.

- drawAimPath(frame, aimpointOffsetArray, aimpoint, frameNr, pathLength)
  - Draws the historical path of the aim-point as a sequence of lines/circles on the frame (maps offsets onto a central aim-point).

- warpFunc(frame, centroid, warpXY)
  - Translates the whole frame so that the detected `centroid` is moved to coordinates `warpXY` — implemented via affine translation (`cv.warpAffine`).

- createVidGraph(...)
  - Builds a large visualization canvas and adds multiple plots (offset, mean, precision, accuracy, Euclidean distance, acceleration). Used to create `graphVid`.

- createFinalVid(...)
  - Assembles the final video frame (polar plot + acceleration + precision + accuracy) placed into the layout produced by `drawFinalLayout`.

---

## Core algorithms and methods
- Template matching
  - Using OpenCV `cv.matchTemplate(img, template, method=cv.TM_CCOEFF_NORMED)`.
  - `cv.minMaxLoc` is used to find the best match location (`max_loc` for TM_CCOEFF_NORMED).

- Numeric measures
  - Offset = aimpoint - centroid (stored as (dx, dy) tuples).
  - Euclidean distance = sqrt(dx^2 + dy^2).
  - Velocity ≈ current_distance - previous_distance (finite difference).
  - Acceleration ≈ current_velocity - previous_velocity (second finite difference).
  - Cumulative mean offset = arithmetic mean across stored offsets.

- Precision & Accuracy (as implemented)
  - A sliding window of the last `checkFrameAmount` frames (default 30) is inspected:
    - Precision: % of points within radius R of the most recent aimpoint (R = 10*(97/40) pixels → treated as "10 cm" with a hard-coded pixels-to-cm factor).
    - Accuracy: % of points within radius R of target center.
  - The code appends those percentages for plotting over time.

- Visualization conversion
  - Matplotlib Figure → FigureCanvasAgg → RGBA buffer → NumPy array → cv.cvtColor(..., COLOR_RGB2BGR) → overlay on OpenCV frame.

---

## Important constants and conversions
- Frame rate assumption: 30 frames per second (used to convert frame index → seconds).
- Pixel-to-centimeter conversion factor: 97/40 (appears repeatedly to convert pixel units into centimeters).
- Precision/accuracy radius: 10 * (97/40) pixels (treated as "10 cm").
- Path length visualization: `pathLength = 10` (number of historical points drawn).

---

## Known issues & suggested fixes
The repository's code is functional but there are several issues and inefficiencies you may want to address:

1. Inefficient frame I/O
   - Current approach writes each frame to disk with `cv.imwrite("frames/frame%d.jpg", frame)` and then reads it back with `cv.imread`. This is slow and unnecessary: process the `frame` directly in memory.

2. Bug: frame counter misuse inside precision/accuracy loop
   - Inside the precision/accuracy computation, the script resets `frameNr = 1` and then uses that value as a local counter. Re-using the same variable name as the outer loop will break the main frame counter. Use a different local variable name (e.g., `local_idx`).

3. Cleanup path mismatch
   - Temporary frames are written to relative path `frames/`, but the final cleanup uses `glob.glob('/frames/*')` (an absolute path). That will not delete the created files. Change to `'frames/*'` or use `os.path.join`.

4. Use of eval to set template matching method
   - `method = eval('cv.TM_CCOEFF_NORMED')` is unnecessary; simply set `method = cv.TM_CCOEFF_NORMED`.

5. Matplotlib overhead
   - Creating a new Figure and Canvas per frame is expensive. Consider:
     - Reusing Matplotlib Figure and Axes objects and updating data (with blitting if necessary).
     - Or use a faster plotting/visualization approach if real-time performance is required.

6. Hard-coded constants
   - `97/40` and the 30 fps assumption should be defined as named constants at the top of the script (e.g., PIXEL_TO_CM and FPS) and documented.

7. Template matching sensitivity
   - Template matching is not scale/rotation-invariant. If the target changes scale or rotates, matching may fail. Consider feature-based matching (ORB/SIFT) or using multi-scale template matching / more robust detectors.

8. Typo/consistency
   - Several variables use non-standard spellings (e.g., `framehight`), which is harmless but may reduce readability.

9. Potential truncation / typing error
   - In some outputs there appeared to be a truncated variable name while calling `createVidGraph` (in the tool output). Ensure the actual repo file is intact—if truncated, it will raise a runtime error.

---

## Performance & optimization suggestions
- Avoid disk round-trips for frames; process frames directly from `cap.read()`.
- Pre-create and reuse Matplotlib Figure/Canvas and only update data for each frame.
- If plotting every frame is not required, downsample plot updates (e.g., update every N frames).
- Use vectorized NumPy operations where possible instead of Python loops for summary statistics.
- If near real-time processing is needed, use optimized libraries (Numba or C++), or move plotting to background threads/processes.

---

## How to extend / ideas
- Support multiple target templates and robust matching (multi-scale, rotation-aware).
- Replace the precision/accuracy thresholds and conversion factors with a configuration file or CLI arguments.
- Save per-frame statistics to CSV or JSON for downstream analysis.
- Add unit tests for numeric computations (offset, velocity, acceleration, precision/accuracy).
- Add CLI flags to toggle which output videos are generated to reduce processing.
