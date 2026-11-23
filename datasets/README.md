# CVAT YOLO Export

This folder contains a local export of CVAT Task 1, Job 1, for `073429_to_074429_10fps.mp4`.

- Export format: Ultralytics YOLO Detection 1.0
- Original export archive: `manual_10min_yolo_detection.zip`
- Extracted dataset: `manual_10min_yolo_detection/`
- Contents verified: 6,000 PNG frames and 5,810 YOLO label files
- Class IDs in `data.yaml`: `0 truck`, `1 car`, `2 bus`, `3 motorcycle`, `4 cyclist`, `5 pedestrian`

The export currently defines `train.txt` only; create a separate validation split before measuring model quality. This format stores per-frame bounding boxes, not persistent tracking IDs. Keep the original ZIP as the untouched CVAT export.

The `datasets/` directory is ignored by Git because it contains a large video-derived dataset. Share it with teammates through an approved data-sharing location if they need the annotations; do not commit it to the public repository.
