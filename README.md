# Traffic Object Tracking

Track road users in a video with YOLOv8 and ByteTrack. The script draws class labels, track IDs, and short trails, and exports detections to CSV.

## Requirements

- Python 3.10 or newer
- pip
- A local video file
- CPU works; a supported CUDA GPU can speed up inference

For an NVIDIA GPU, install the PyTorch build matching the team's operating system, Python version, and CUDA option using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install this project's requirements. Verify GPU access with `python -c "import torch; print(torch.cuda.is_available())"`; it should print `True` to use CUDA.

Install dependencies from PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Download the team video, [Video of people across the road](https://drive.google.com/file/d/1-KLIsu6hHfYMYuT4ZT1Hj8XNRZnDMJ6E/view?usp=sharing), from Google Drive and save it locally. Each team member needs access to the shared Drive file. The script needs the local file path; it cannot read the Drive page URL directly. The first run downloads the model weights automatically if they are not present.

```powershell
python track.py --source "D:\Research Intern - Connected Urban Mobility\Video of people across the road.mp4" --seconds 30 --output outputs_test
```

Process the first three minutes at higher image resolution:

```powershell
python track.py --source "D:\Videos\traffic.mp4" --seconds 180 --imgsz 1280 --conf 0.05 --output outputs_run
```

## Run the strongest released YOLO detector

Ultralytics lists YOLO26 as its latest released family. `yolo26x.pt` is its largest detection model and the accuracy-first option in that family. It needs more GPU memory and runs more slowly than the `n`, `s`, or `m` models. For practical use, run it on a CUDA-enabled NVIDIA GPU; this project's current authoring machine has CPU-only PyTorch, so running the `x` model here is not recommended.

After installing the environment and placing the source video on the machine, use this command to process only the requested video, for three minutes, into a dedicated output folder:

```powershell
python track.py --source "D:\Research Intern - Connected Urban Mobility\Video of people across the road.mp4" --seconds 180 --model yolo26x.pt --imgsz 1280 --conf 0.05 --output outputs_people_yolo26x_3min
```

Ultralytics downloads `yolo26x.pt` automatically on the first run. The video path is local to each team member and is intentionally excluded from Git. If the GPU runs out of memory, lower `--imgsz` to `960`; use `yolo26l.pt` or `yolo26m.pt` if the `x` model is too slow or too large for the available GPU.

The default sampling stride is 3. For a 30 FPS source, the script processes 10 frames per second of video and preserves the source duration in the output. Increase `--stride` for faster, less dense tracking; use `--stride 1` to process every frame. The output video is resized to at most 1280 pixels wide and has no audio.

## Options

| Option | Default | Purpose |
|---|---:|---|
| `--source` | required | Input video path |
| `--seconds` | 180 | Maximum source duration to process |
| `--stride` | 3 | Process every Nth frame |
| `--imgsz` | 960 | YOLO inference image size |
| `--conf` | 0.05 | Minimum detection confidence |
| `--model` | `yolov8s.pt` | YOLO checkpoint; for example `yolov8m.pt` |
| `--output` | `outputs/` | Directory for generated files |

Lower `--conf` can recover small or distant objects, but may add false detections. Higher `--imgsz` can help with small objects and uses more time and memory. Larger models such as `yolov8m.pt` may improve recognition but run more slowly. Change one option at a time and compare the same video segment.

ByteTrack settings are in `bytetrack.yaml`. `track_high_thresh`, `track_low_thresh`, and `new_track_thresh` affect matching and when new IDs are created. Lower thresholds can keep more weak detections, but may also create unstable or false tracks.

## Output files

- `tracking_3min.mp4`: annotated video (the filename is fixed, even when processing a shorter clip).
- `tracks.csv`: source frame, time, track ID, class, confidence, and bounding-box coordinates in the original frame.
- `summary.json`: run settings and aggregate counts.
- `preview_*.jpg`: selected annotated frames for a quick visual check.

Use a different `--output` directory for each run. Otherwise, files with the same names are replaced.

## Classes and limitations

The model's COCO `person`, `car`, `bus`, `truck`, and `motorcycle` classes are displayed as `pedestrian`, `car`, `bus`, `truck`, and `scooter`. The `scooter` label is an approximation: COCO's motorcycle class can also include motorcycles, so it cannot reliably distinguish an electric scooter from a motorbike.

COCO does not provide a dedicated cyclist class. This project approximates `cyclist` by associating a detected person with a nearby bicycle. The heuristic can miss distant or occluded riders and can mistake someone walking a bicycle for a cyclist. A custom model trained on representative, labeled frames is needed for reliable distinctions.

Track IDs can switch or fragment during occlusion. Counts in `summary.json` are detection observations and unique tracker IDs, not verified counts of physical people or vehicles. Evaluate precision and recall against manually labeled frames before relying on the results.

## GitHub collaboration

Input videos, downloaded model weights, virtual environments, and generated output folders are excluded by `.gitignore`. Share code and configuration through GitHub, and have each collaborator provide their own local video path with `--source`.

```powershell
git clone <repository-url>
cd <repository-folder>
python -m pip install -r requirements.txt
python track.py --source "D:\Videos\traffic.mp4" --seconds 30 --output outputs_test
```

## References

- [Ultralytics tracking mode](https://docs.ultralytics.com/modes/track/)
- [ByteTrack paper](https://arxiv.org/abs/2110.06864)
