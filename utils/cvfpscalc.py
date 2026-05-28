from collections import deque
from time import perf_counter


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_time = perf_counter()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_time = perf_counter()
        different_time = (current_time - self._start_time) * 1000.0
        self._start_time = current_time

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
