"""Bounded read-ahead and write-behind for the per-frame image loops.

The mask stage is mostly codec work around a short GPU burst: decoding 3840x3840
fisheye JPEGs and 7680x3840 panoramas, encoding PNG masks. OpenCV's imread,
imwrite and resize release the GIL, so a few threads run them in parallel while
the main thread keeps the GPU busy. Measured on clip 0005 (957 rig frames),
the loops were serial and a 46-core host sat at 3 busy cores with the GPU
idle most of the time.

Order is preserved on both sides, so every output and every running total is
computed exactly as the serial loop computed it. Both sides are bounded,
because a producer faster than the writer would otherwise queue whole
panoramas (88 MB each at 7680x3840) until the host runs out of memory.

The thread count follows SPLAT_THREADS (the queue sets it to the container's
CPU allowance), capped: past about 8 threads the disk, not the codec, is the
limit.
"""
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

_END = object()


def io_threads(cap: int = 8) -> int:
    n = int(os.environ.get("SPLAT_THREADS") or 0) or os.cpu_count() or 1
    return max(1, min(cap, n))


def read_ahead(fn, items, workers=None, depth=None):
    """Yield (item, fn(item)) in input order, keeping up to `depth` calls in flight."""
    workers = workers or io_threads()
    depth = depth or 2 * workers
    with ThreadPoolExecutor(workers) as ex:
        it = iter(items)
        q = deque()
        while True:
            while len(q) < depth:
                x = next(it, _END)
                if x is _END:
                    break
                q.append((x, ex.submit(fn, x)))
            if not q:
                return
            x, fut = q.popleft()
            yield x, fut.result()


class WriteBehind:
    """Run writes on a thread pool with at most `depth` outstanding.

    A write that returns False or raises fails the run at the next submit or at
    close(), never silently: cv2.imwrite reports a full disk by returning False.
    """

    def __init__(self, workers=None, depth=None):
        workers = workers or io_threads()
        self.depth = depth or 2 * workers
        self.ex = ThreadPoolExecutor(workers)
        self.q = deque()

    def submit(self, what, fn, *args):
        self.q.append((what, self.ex.submit(fn, *args)))
        while len(self.q) > self.depth:
            self._check(*self.q.popleft())

    @staticmethod
    def _check(what, fut):
        if fut.result() is False:
            raise SystemExit(f"could not write {what}")

    def close(self):
        try:
            while self.q:
                self._check(*self.q.popleft())
        finally:
            self.ex.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:                           # already failing: do not mask the error
            self.q.clear()
            self.ex.shutdown(wait=True, cancel_futures=True)
