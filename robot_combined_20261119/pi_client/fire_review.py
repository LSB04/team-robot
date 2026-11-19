"""Stationary fire review for navigation only; never authorizes a shot."""
from collections import deque
from statistics import median

VERSION = 'fire_stationary_review_v4_target_identity'

# Navigation confirmation only. LauncherFireReview and firing thresholds are separate.
PERSISTENT_WINDOW = 10
PERSISTENT_HITS = 8
PERSISTENT_MIN_SECONDS = 1.2
PERSISTENT_MAX_SECONDS = 3.0


class FireReview:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.engaged = self.confirmed = False
        self.seq = None
        self.box = self.current = None
        self.hold_at = None
        self.last_capture = None
        self.last_seen_capture = None
        self.evidence = deque(maxlen=5)
        self.persistent = deque(maxlen=PERSISTENT_WINDOW)
        self.confirmation_path = None
        self.target_id = 0
        self.association = 'none'

    def nearby_target(self, obj, capture):
        # Identity only: a displaced box still has to pass stopped confirmation.
        # Do not reset a completed approach phase just because a small target
        # moves by one box width as the chassis settles.
        if self.box is None or self.last_seen_capture is None:
            return False
        if not 0 <= capture-self.last_seen_capture <= 3.:
            return False
        a, b = self.box, obj['box']
        aw, ah, bw, bh = a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1]
        if min(aw, ah, bw, bh) <= 0:
            return False
        return (.5 <= bw/aw <= 2. and .5 <= bh/ah <= 2.
                and abs((b[0]+b[2]-a[0]-a[2])/2) <= 1.25*max(aw,bw)
                and abs((b[1]+b[3]-a[1]-a[3])/2) <= max(ah,bh))

    def candidates(self, vision):
        return [d for d in (vision or {}).get('detections', [])
                if d['label'] == 'fire'
                and d['confidence'] >= self.cfg.get('fire_candidate_conf', .35)]

    def visible_match(self, vision, overlap):
        return any(self.box is not None and overlap(self.box, d['box']) >= .2
                   for d in self.candidates(vision))

    def pause(self, now):
        self.confirmed = False
        self.current = None
        self.hold_at = now
        self.evidence.clear()
        self.persistent.clear()
        self.confirmation_path = None

    def observe(self, capture, score):
        self.evidence.append(score)
        self.persistent.append((capture, score))
        while self.persistent and capture-self.persistent[0][0] > PERSISTENT_MAX_SECONDS:
            self.persistent.popleft()

    def persistent_progress(self):
        hits = [t for t, score in self.persistent if score > 0]
        return len(hits), (hits[-1]-hits[0] if len(hits)>1 else 0.)

    def update(self, vision, now, overlap):
        if vision['seq'] == self.seq:
            return self.current if self.confirmed else None
        self.seq = vision['seq']
        capture = vision['capture_mono']
        candidates = self.candidates(vision)
        matches = [d for d in candidates if self.box is not None and overlap(self.box, d['box']) >= .2]
        nearby = [d for d in candidates if self.nearby_target(d, capture)]
        # Ambiguous nearby detections must not inherit an earlier route.
        pool = matches or (nearby if len(nearby) == 1 else candidates)
        obj = max(pool, key=lambda d: d['confidence'], default=None)
        same = obj is not None and self.box is not None and overlap(self.box, obj['box']) >= .2
        retained = same or (obj is not None and len(nearby) == 1 and obj is nearby[0])
        if self.last_capture is not None and (capture <= self.last_capture or capture-self.last_capture > .8):
            self.pause(now)
        self.last_capture = capture
        if obj is None:
            if self.confirmed:
                self.pause(now)
            self.current = None
            if self.engaged and self.hold_at is not None and capture >= self.hold_at+.5:
                self.observe(capture, 0.)
            return None
        self.engaged = True
        if not retained:
            self.target_id += 1
        self.association = 'overlap' if same else 'nearby_reconfirm' if retained else 'new_target'
        if not same or self.hold_at is None:
            self.pause(now)
        self.box = list(obj['box'])
        self.last_seen_capture = capture
        self.current = obj
        if self.confirmed:
            return obj
        # Moving/settling frames must never confirm a stopped observation.
        if capture >= self.hold_at+.5:
            self.observe(capture, obj['confidence'])
        scores = list(self.evidence)
        if sum(s > 0 for s in scores) >= 3 and max(scores, default=0.) >= self.cfg.get('fire_review_peak_conf', .45):
            self.confirmed = True
            self.confirmation_path = 'peak_3_of_5'
            return obj
        hits, span = self.persistent_progress()
        if hits >= PERSISTENT_HITS and span >= PERSISTENT_MIN_SECONDS:
            self.confirmed = True
            self.confirmation_path = 'persistent_8_of_10'
            return obj
        return None

    def diagnostic(self):
        center = None if self.box is None else dict(x_px=(self.box[0]+self.box[2])/2,
            y_px=(self.box[1]+self.box[3])/2,last_seen_capture_mono=self.last_seen_capture,
            currently_visible=self.current is not None,coordinate_system='camera_image_pixels')
        hits, span = self.persistent_progress()
        return dict(version=VERSION, engaged=self.engaged, confirmed=self.confirmed,
                    target_id=self.target_id, association=self.association,
                    confirmation_path=self.confirmation_path,
                    peak_required=self.cfg.get('fire_review_peak_conf', .45),
                    persistent_hits=hits, persistent_window=len(self.persistent),
                    persistent_hits_required=PERSISTENT_HITS, persistent_window_limit=PERSISTENT_WINDOW,
                    persistent_span_seconds=span, persistent_min_seconds=PERSISTENT_MIN_SECONDS,
                    hits=sum(x > 0 for x in self.evidence), window=len(self.evidence),
                    peak=max(self.evidence, default=0.),
                    visible_score=None if self.current is None else self.current['confidence'],
                    box=self.box, center=center)


class FireRangeReview:
    """Fresh distinct samples while stopped. Never choose the farther cluster."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.stopped_at = None
        self.last_seq = None
        self.samples = deque(maxlen=5)

    def update(self, now, distance, sample_id, was_moving):
        if was_moving or self.stopped_at is None:
            self.reset()
            self.stopped_at = now
            self.last_seq = sample_id
        if distance is None:
            self.samples.clear()
            return None
        if now-self.stopped_at < .3 or sample_id == self.last_seq:
            return None
        self.last_seq = sample_id
        self.samples.append((now, distance))
        recent = [d for t,d in self.samples if now-t <= 1.2]
        if len(recent) < 5 or max(recent)-min(recent) > 20.:
            return None
        # Straddling the approach boundary requires more stopped readings.
        return recent

    def diagnostic(self):
        values=[d for _,d in self.samples]
        return dict(samples=len(values), median_cm=median(values) if values else None,
                    spread_cm=max(values)-min(values) if values else None)
