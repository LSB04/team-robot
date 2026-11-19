"""Short fresh-sample median for stationary launcher aiming only."""
from collections import deque
from statistics import median
from core import valid_distance

class LauncherRange:
    def __init__(self): self.reset()
    def reset(self):
        self.samples=deque(maxlen=9); self.seq=None; self.value=None
    def update(self,now,distance,seq):
        if seq != self.seq:
            self.seq=seq
            if valid_distance(distance): self.samples.append((now,float(distance)))
        while self.samples and now-self.samples[0][0]>1.0: self.samples.popleft()
        self.value=None
        if not valid_distance(distance) or len(self.samples)<5 or now-self.samples[-1][0]>.3: return None
        values=[v for _,v in self.samples]; middle=median(values)
        agreeing=[v for v in values if abs(v-middle)<=15.]
        if len(agreeing)>=5: self.value=median(agreeing)
        return self.value
    def diagnostic(self):
        return dict(method='fresh_9_sample_majority_median',samples=len(self.samples),distance_cm=self.value)
