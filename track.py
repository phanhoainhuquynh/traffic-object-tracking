from pathlib import Path
import argparse, csv, json, time, math
from collections import defaultdict, deque, Counter
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
COLORS = {'car':(80,220,80),'pedestrian':(255,190,50),'cyclist':(40,220,255),'bus':(255,100,200),'truck':(80,140,255),'scooter':(200,120,255)}

def visible_objects(boxes):
    """Infer cyclists by geometric person+bicycle association; not a trained cyclist class."""
    rows = boxes.cpu().numpy().data if boxes.is_track else np.empty((0,7))
    people = [r for r in rows if int(r[6]) == 0]
    used = set()
    output = []
    for b in rows:
        if int(b[6]) != 1: continue
        bw,bh = b[2]-b[0], b[3]-b[1]
        candidates=[]
        for p in people:
            if int(p[4]) in used: continue
            px=(p[0]+p[2])/2
            overlap=max(0,min(p[2],b[2])-max(p[0],b[0]))/max(1,min(p[2]-p[0],bw))
            if overlap>.35 and b[0]-.1*bw<px<b[2]+.1*bw and p[1]<b[1]+.4*bh and b[1]-.2*bh<p[3]<b[3]+.3*bh:
                candidates.append((abs(px-(b[0]+b[2])/2),p))
        if candidates:
            p=min(candidates,key=lambda x:x[0])[1]; used.add(int(p[4]))
            output.append((int(b[4]),'cyclist',min(float(b[5]),float(p[5])),[min(b[0],p[0]),min(b[1],p[1]),max(b[2],p[2]),max(b[3],p[3])]))
    mapping={0:'pedestrian',2:'car',3:'scooter',5:'bus',7:'truck'}
    for r in rows:
        cls=int(r[6]); tid=int(r[4])
        if cls in mapping and tid not in used:
            output.append((tid,mapping[cls],float(r[5]),r[:4].tolist()))
    return output

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True,help='Path to the input video file.')
    ap.add_argument('--seconds',type=float,default=180)
    ap.add_argument('--stride',type=int,default=3)
    ap.add_argument('--imgsz',type=int,default=960)
    ap.add_argument('--conf',type=float,default=0.05,help='Detection confidence threshold (0-1). Lower finds more objects but adds false positives.')
    ap.add_argument('--model',default=str(ROOT/'yolov8s.pt'))
    ap.add_argument('--output',type=Path,default=ROOT/'outputs')
    args=ap.parse_args()
    if args.seconds<=0 or args.stride<1 or not 0<args.conf<1: ap.error('seconds > 0, stride >= 1 and 0 < conf < 1 required')
    args.output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(args.source))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {args.source}')
    fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0: raise RuntimeError('Invalid source FPS')
    limit=min(total,math.floor(args.seconds*fps)); outfps=fps/args.stride
    model=YOLO(args.model)
    writer=None; trails=defaultdict(lambda:deque(maxlen=25)); counts=Counter(); ids=defaultdict(set)
    start=time.perf_counter(); processed=0
    try:
        with (args.output/'tracks.csv').open('w',newline='',encoding='utf-8') as file:
            csvout=csv.writer(file); csvout.writerow(['source_frame','time_seconds','track_id','class','confidence','x1','y1','x2','y2'])
            for index in range(limit):
                if not cap.grab(): break
                if index%args.stride: continue
                ok,frame=cap.retrieve()
                if not ok: break
                result=model.track(frame,persist=True,tracker=str(ROOT/'bytetrack.yaml'),conf=args.conf,iou=.7,imgsz=args.imgsz,classes=[0,1,2,3,5,7],verbose=False)[0]
                objects=visible_objects(result.boxes)
                h,w=frame.shape[:2]; scale=min(1,1280/w)
                canvas=cv2.resize(frame,(round(w*scale),round(h*scale)))
                for tid,label,conf,xy in objects:
                    csvout.writerow([index,round(index/fps,4),tid,label,round(conf,4),*[round(float(v),2) for v in xy]])
                    counts[label]+=1; ids[label].add(tid)
                    x1,y1,x2,y2=[int(v*scale) for v in xy]; color=COLORS[label]
                    trails[tid].append(((x1+x2)//2,y2))
                    cv2.polylines(canvas,[np.array(trails[tid],np.int32)],False,color,2)
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                    text=f'{label} #{tid} {conf:.2f}'
                    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.5,1)
                    yy=max(th+5,y1)
                    cv2.rectangle(canvas,(x1,yy-th-5),(min(canvas.shape[1],x1+tw+4),yy),color,-1)
                    cv2.putText(canvas,text,(x1+2,yy-3),cv2.FONT_HERSHEY_SIMPLEX,.5,(15,15,15),1,cv2.LINE_AA)
                cv2.rectangle(canvas,(0,0),(canvas.shape[1],35),(25,25,25),-1)
                cv2.putText(canvas,f'Time {index/fps:6.1f}s | Active: {len(objects)} | YOLOv8s + ByteTrack',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
                if writer is None:
                    writer=cv2.VideoWriter(str(args.output/'tracking_3min.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),outfps,(canvas.shape[1],canvas.shape[0]))
                    if not writer.isOpened(): raise RuntimeError('Cannot open video writer')
                writer.write(canvas); processed+=1
                if processed in [1,101,301,601,1201,1701]: cv2.imwrite(str(args.output/f'preview_{processed:04d}.jpg'),canvas)
                if processed%100==0: print(f'Processed {processed} frames, source {index/fps:.1f}s, elapsed {time.perf_counter()-start:.1f}s',flush=True)
                if processed%100==0:
                    active={t[0] for t in objects}
                    for tid in list(trails):
                        if tid not in active: del trails[tid]
    finally:
        cap.release()
        if writer is not None: writer.release()
    summary={'source':str(args.source),'source_fps':fps,'source_frames':total,'processed_frames':processed,'output_fps':outfps,'output_duration_seconds':processed/outfps,'elapsed_seconds':round(time.perf_counter()-start,2),'model':args.model,'imgsz':args.imgsz,'confidence_threshold':args.conf,'stride':args.stride,'observations_by_class':dict(counts),'unique_ids_by_class':{k:len(v) for k,v in ids.items()},'cyclist_method':'Geometric person+bicycle pairing, bicycle track ID. Unpaired bicycles omitted; pairing may confuse people walking a bicycle.','limitations':'IDs may switch or fragment; counts are not unique physical objects. No ground-truth accuracy evaluation.'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()

