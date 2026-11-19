"""Whole-frame inference, launcher detail, and fresh-frame fire reacquisition."""

REACQUIRE_VERSION = 'remembered_fire_roi_v1'


class RememberedFireReview:
    """Remember where to LOOK, never generate a detection from old coordinates."""
    MAX_AGE_SECONDS = 3.0

    def __init__(self):
        self.clear()

    def clear(self):
        self.box = self.last_seen = self.last_capture = self.last_seq = self.camera_key = None

    def predict(self, detector, frame, capture_mono, frame_seq, camera_key=None, fire_threshold=.35):
        height, width = frame.shape[:2]
        key = (width, height, camera_key)
        if (key != self.camera_key or (self.last_capture is not None
                and (capture_mono <= self.last_capture or frame_seq == self.last_seq))):
            self.clear()
        self.camera_key, self.last_capture, self.last_seq = key, capture_mono, frame_seq
        if self.last_seen is not None and capture_mono-self.last_seen > self.MAX_AGE_SECONDS:
            self.box = self.last_seen = None
        whole, elapsed, info = predict_review(detector, frame, fire_threshold=fire_threshold)
        info.update(reacquire_version=REACQUIRE_VERSION, remembered_box=self.box,
                    reacquire_attempted=False, reacquired=False)
        fires = [d for d in whole if d['label']=='fire']
        matches = [d for d in fires if self.box and overlap(self.box, d['box']) >= .2]
        merged = list(whole)
        if self.box is not None and not matches:
            cx, cy = (self.box[0]+self.box[2])/2, (self.box[1]+self.box[3])/2
            bw, bh = self.box[2]-self.box[0], self.box[3]-self.box[1]
            # Expand a bounded neighborhood in the CURRENT raw camera image.
            rw = min(width, round(min(width*.65, max(width*.4, bw*6))))
            rh = min(height, round(min(height*.65, max(height*.4, bh*6))))
            x = max(0, min(width-rw, round(cx-rw/2)))
            y = max(0, min(height-rh, round(cy-rh/2)))
            extra, cost = detector.predict(frame[y:y+rh, x:x+rw])
            elapsed += cost
            info.update(passes=2, roi=[x,y,rw,rh], reacquire_attempted=True)
            for obj in extra:
                if obj['label'] not in ('fire','person'):
                    continue
                a,b,c,d = obj['box']
                mapped = dict(obj, box=[a+x,b+y,c+x,d+y], inference_source='remembered_detail')
                if obj['label']=='fire':
                    if (obj['confidence'] < fire_threshold or min(a,b,rw-c,rh-d) < 2
                            or overlap(self.box, mapped['box']) < .2):
                        continue
                    info['reacquired'] = True
                duplicate = next((i for i, old in enumerate(merged)
                    if old['label']==mapped['label'] and overlap(old['box'],mapped['box'])>=.45), None)
                if duplicate is None:
                    merged.append(mapped)
                elif mapped['label']=='fire' and mapped['confidence']>merged[duplicate]['confidence']:
                    merged[duplicate] = mapped
        fires = [d for d in merged if d['label']=='fire']
        matches = [d for d in fires if self.box and overlap(self.box,d['box'])>=.2]
        selected = max(matches or fires, key=lambda d:d['confidence'], default=None)
        if selected is not None:
            self.box, self.last_seen = list(selected['box']), capture_mono
        merged.sort(key=lambda d:(d['label']!='person',-d['confidence']))
        return merged[:100], elapsed, info


def overlap(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area > 0 else 0.


def predict_review(detector, frame, detail=False, fire_threshold=.35):
    objects, elapsed = detector.predict(frame)
    whole = [dict(d, inference_source='full') for d in objects
             if d['label']!='fire' or d['confidence']>=fire_threshold]
    info = dict(passes=1, model_input_size=detector.size, roi=None)
    if not detail:
        return whole, elapsed, info
    height, width = frame.shape[:2]
    # Fire targets above this low-mounted camera tend to occupy the upper view.
    # The first pass still covers every pixel, including outside this ROI.
    x, y = round(width*.2), round(height*.05)
    rw, rh = round(width*.6), round(height*.6)
    extra, cost = detector.predict(frame[y:y+rh, x:x+rw])
    # Whole-frame people are always kept. The second pass also contributes
    # people so a cropped detail view can never hide them from priority checks.
    merged = list(whole)
    for obj in extra:
        if obj['label'] not in ('fire', 'person'):
            continue
        if obj['label']=='fire' and obj['confidence']<fire_threshold:
            continue
        a,b,c,d = obj['box']
        # A cropped fire edge is not a reliable target center.
        if obj['label'] == 'fire' and min(a,b,rw-c,rh-d) < 2:
            continue
        mapped = dict(obj, box=[a+x,b+y,c+x,d+y], inference_source='center_detail')
        # Preserve whole-frame person boxes even if detail has higher confidence.
        match = next((i for i,old in enumerate(merged) if old['label']==mapped['label']
                      and overlap(old['box'],mapped['box'])>=.45), None)
        if match is None:
            merged.append(mapped)
        elif mapped['label']=='fire' and mapped['confidence']>merged[match]['confidence']:
            merged[match] = mapped
    merged.sort(key=lambda obj:(obj['label']!='person',-obj['confidence']))
    info.update(passes=2,roi=[x,y,rw,rh])
    return merged[:100], elapsed+cost, info
