"""Count unique manually annotated F1 tracks in CVAT job 1 (720x480, 10 fps)."""
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs_crossings"
LINE = ((468, 479), (609, 264))
CLASSES = ("pedestrian", "cyclist")
VEHICLES = ("car", "truck", "bus")


# OpenCV uses blue, green, red channel order.
CLASS_COLORS = {
    "car": (245, 145, 45),
    "truck": (25, 135, 255),
    "bus": (205, 80, 195),
    "pedestrian": (95, 230, 70),
    "cyclist": (75, 65, 250),
}

def visible_boxes(track, stop):
    shapes = sorted(track["shapes"], key=lambda s: s["frame"])
    for i, shape in enumerate(shapes):
        if shape["outside"]:
            continue
        following = shapes[i+1] if i+1 < len(shapes) else None
        first = shape["frame"]
        end = following["frame"] if following else stop
        start_box = np.array(shape["points"])
        end_box = np.array(following["points"]) if following else start_box
        for frame in range(first, min(stop, end)):
            box = start_box + (end_box-start_box) * ((frame-first)/(end-first))
            yield frame, tuple(np.rint(box).astype(int).tolist())


def main():
    data = json.loads((OUT/"tracks.json").read_text(encoding="utf-8-sig"))
    cap = cv2.VideoCapture(str(OUT/"source.mp4"))
    stop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (cap.get(3), cap.get(4), cap.get(5), stop) != (720, 480, 10, 6000):
        raise ValueError("Expected the annotated 10-minute clip at 720x480 and 10 fps")
    boxes = defaultdict(list)
    for track in data["tracks"]:
        label = data["labels"][str(track["label_id"])]
        if label in VEHICLES or label in CLASSES and any(a["value"] == "F1" for a in track["attributes"]):
            for frame, box in visible_boxes(track, stop):
                boxes[frame].append((track["id"], label, box))
    seen, touched = set(), set()
    totals, hits = Counter(), Counter()
    events = []
    output = OUT/"F1_count.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", "720x480", "-r", "10", "-i", "-", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for frame in range(stop):
                ok, image = cap.read()
                if not ok:
                    raise IOError(f"Could not read frame {frame}")
                cv2.line(image, *LINE, (0,255,255), 2)
                for tid, label, (x,y,r,b) in boxes[frame]:
                    color = CLASS_COLORS[label]
                    if label in CLASSES:
                        contact = cv2.clipLine((x,y,max(1,r-x+1),max(1,b-y+1)), *LINE)[0]
                        if contact:
                            color = (0, 255, 255)
                        for kind, active, ids, counts in (("F1",True,seen,totals), ("line",contact,touched,hits)):
                            if active and tid not in ids:
                                ids.add(tid)
                                counts[label] += 1
                                events.append((frame,frame/10,kind,label,tid,counts[label]))

                    cv2.rectangle(image,(x,y),(r,b),color,2)
                    cv2.putText(image,f"{label} #{tid}",(max(0,x),max(85,y-3)),0,0.43,(15,15,15),3,cv2.LINE_AA)
                    cv2.putText(image,f"{label} #{tid}",(max(0,x),max(85,y-3)),0,0.43,color,1,cv2.LINE_AA)
                cv2.rectangle(image,(0,0),(420,65),(25,25,25),-1)
                for row,(name,counts) in enumerate((("F1",totals),("F1 line",hits))):
                    cv2.putText(image,f"{name}   Pedestrian: {counts['pedestrian']}   Cyclist: {counts['cyclist']}",(8,24+28*row),0,.63,(255,255,255),1,cv2.LINE_AA)
                encoder.stdin.write(image.tobytes())
        finally:
            encoder.stdin.close()
            cap.release()
        if encoder.wait():
            raise RuntimeError("Video encoding failed")
    with (OUT/"F1_events.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("frame_zero_based","seconds","event","class","cvat_track_id","class_total"))
        writer.writerows(events)
    print(totals, hits)


if __name__ == "__main__":
    main()




