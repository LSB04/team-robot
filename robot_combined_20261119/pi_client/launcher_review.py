"""Recent same-target evidence for stationary aiming; never reuse a missing box."""
from collections import deque

VERSION = 'launcher_review_v2_configurable_threshold'


class LauncherFireReview:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.seq = self.last_capture = None
        self.box = self.current = None
        self.evidence = deque(maxlen=5)
        self.confirmed = False

    def candidates(self, vision):
        return [d for d in (vision or {}).get('detections', [])
                if d['label'] == 'fire'
                and d['confidence'] >= self.cfg.get('launcher_candidate_conf', .37)]

    def update(self, vision, overlap):
        if vision['seq'] == self.seq:
            return self.current
        capture = vision['capture_mono']
        if self.last_capture is not None and capture <= self.last_capture:
            self.reset()
            return None
        if self.last_capture is not None and capture-self.last_capture > .8:
            self.reset()
        self.seq, self.last_capture = vision['seq'], capture
        candidates = self.candidates(vision)
        matches = [d for d in candidates if self.box is not None and overlap(self.box, d['box']) >= .3]
        obj = max(matches or candidates, key=lambda d: d['confidence'], default=None)
        if obj is not None:
            if self.box is None or overlap(self.box, obj['box']) < .3:
                self.evidence.clear()
            self.box = list(obj['box'])
        self.current = obj
        self.evidence.append((capture, obj['confidence'] if obj else 0.))
        # A gap may retain tracking evidence, but never supplies a current target.
        while self.evidence and capture-self.evidence[0][0] > 1.2:
            self.evidence.popleft()
        scores = [s for _, s in self.evidence]
        self.confirmed = (obj is not None and sum(s > 0 for s in scores) >= 3
                          and max(scores, default=0.) >= self.cfg.get('launcher_review_peak_conf', .37))
        return obj

    def diagnostic(self):
        scores = [s for _, s in self.evidence]
        return dict(version=VERSION, confirmed=self.confirmed, hits=sum(s > 0 for s in scores),
                    window=len(scores), peak=max(scores, default=0.), box=self.box,
                    visible_score=None if self.current is None else self.current['confidence'],
                    candidate_conf=self.cfg.get('launcher_candidate_conf', .37),
                    confirmation_peak_conf=self.cfg.get('launcher_review_peak_conf', .37),
                    firing_conf=self.cfg['fire_conf'])
