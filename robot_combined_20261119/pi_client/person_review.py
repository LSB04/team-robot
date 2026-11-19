"""Repeated low-confidence person candidates require stationary fresh frames."""
class PersonReview:
    def __init__(self, hold_while_visible=False):
        self.hold_while_visible = hold_while_visible
        self.reset()

    def reset(self):
        self.seq = -1
        self.box = None
        self.count = self.confirm = 0
        self.hold_at = None
        self.last_seen = -1e9
        self.last_capture = None

    def update(self, vision, now, overlap):
        if vision['seq'] == self.seq:
            return 'hold' if self.hold_at is not None else None
        if self.hold_while_visible and self.last_capture is not None:
            if vision['capture_mono'] <= self.last_capture:
                return 'hold' if self.hold_at is not None else None
            if vision['capture_mono']-self.last_capture > .7:
                self.count = self.confirm = 0
        self.last_capture = vision['capture_mono']
        self.seq = vision['seq']
        candidates = [d for d in vision['detections'] if d['label']=='person' and d['confidence']>=.35]
        obj = max(candidates,key=lambda d:d['confidence']) if candidates else None
        same = obj is not None and self.box is not None and overlap(self.box,obj['box'])>=.3
        if obj:
            self.count = self.count+1 if same and now-self.last_seen<=.7 else 1
            self.box, self.last_seen = obj['box'],now
        else:
            self.count = 0
        if self.hold_at is None:
            if self.count>=3:
                self.hold_at=now
                self.confirm=0
                return 'hold'
            return None
        # Only frames captured after settling count as stopped re-observations.
        if vision['capture_mono']>=self.hold_at+.5:
            self.confirm = self.confirm+1 if obj and same else 0
        if self.confirm>=5:
            return 'confirmed'
        if now-self.last_seen>=1.5 or (not self.hold_while_visible and now-self.hold_at>=5):
            self.reset()
            return 'clear'
        return 'hold'


class ForwardBalance:
    """Bounded differential stride. Base amplitude is unchanged."""
    def __init__(self):
        self.target = self.filtered = None
        self.boost = 0.

    def update(self, active, yaw):
        if not active:
            self.target=self.filtered=None
            self.boost=0.
            return 0.
        if self.target is None:
            self.target=self.filtered=yaw
        self.filtered += .2*(yaw-self.filtered)
        error=self.target-self.filtered
        desired = 0. if abs(error)<=5 else max(-.06,min(.06,error*.004))
        self.boost += max(-.01,min(.01,desired-self.boost))
        return self.boost
