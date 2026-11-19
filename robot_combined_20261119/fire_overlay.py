"""Image-space fire center memory for display/logging, never motion control."""
def overlap(a, b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union>0 else 0.


class FireCenterMemory:
    def __init__(self, threshold=.35):
        self.threshold=threshold
        self.clear()

    def clear(self):
        self.last=None

    def update(self, vision, now, selected_box=None):
        candidates=[d for d in vision.get('detections',[])
                    if d['label']=='fire' and d['confidence']>=self.threshold]
        if selected_box is not None:
            # A fresh control plan is authoritative for the displayed aim point.
            candidates=[d for d in candidates if overlap(selected_box,d['box'])>=.2]
        matches=[d for d in candidates if self.last and overlap(self.last['box'],d['box'])>=.2]
        obj=max(matches or candidates,key=lambda d:d['confidence'],default=None)
        if obj:
            box=list(selected_box if selected_box is not None else obj['box'])
            self.last=dict(x_px=(box[0]+box[2])/2,y_px=(box[1]+box[3])/2,box=box,
                           confidence=obj['confidence'],source_seq=vision['seq'],
                           capture_mono=vision['capture_mono'],last_seen_pc_monotonic=now,
                           coordinate_system='camera_image_pixels_not_world_coordinates',
                           source='control_plan' if selected_box is not None else 'vision_detection')
        if self.last is None:
            return None
        return dict(self.last,observed=obj is not None,age_seconds=max(0.,now-self.last['last_seen_pc_monotonic']))

    @staticmethod
    def draw(frame, center):
        if center is None:
            return
        import cv2
        height,width=frame.shape[:2]
        x=max(0,min(width-1,round(center['x_px'])))
        y=max(0,min(height-1,round(center['y_px'])))
        observed=center['observed']
        color=(255,220,40) if observed else (170,170,170)
        label=f'F ({x}, {y})' if observed else f'LAST ({x}, {y}) {center["age_seconds"]:.1f}s'
        if observed:
            cv2.drawMarker(frame,(x,y),(0,0,0),cv2.MARKER_CROSS,19,4)
            cv2.drawMarker(frame,(x,y),color,cv2.MARKER_CROSS,19,2)
            cv2.circle(frame,(x,y),4,color,1)
        else:
            for a,b in [((x-9,y),(x-3,y)),((x+3,y),(x+9,y)),((x,y-9),(x,y-3)),((x,y+3),(x,y+9))]:
                cv2.line(frame,a,b,color,2)
        font_scale = .7 if width >= 960 else .45
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,font_scale,1)
        tx=max(2,min(width-tw-2,x+12));ty=max(th+3,min(height-4,y-12))
        cv2.putText(frame,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(frame,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,2,cv2.LINE_AA)
