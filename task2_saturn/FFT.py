import numpy as np
import time
import cv2

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def ft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:
        return ft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        n = int(N / 2)
        return np.concatenate([X_even + factor[:n] * X_odd,
                               X_even + factor[n:] * X_odd])

@timeit
def fft_timed(x):
    return fft(x)

@timeit
def fft2_timed(x):
    return np.asarray([
        fft(col) for col in
        zip(*[fft(row) for row in x])
    ])


if __name__ == '__main__':
    x = cv2.imread('../data/saturn.jpg', 0)
    result_my = np.abs(fft2_timed(x))
    cv2.imshow('fft2_my', result_my * 255 / np.max(result_my))
    cv2.waitKey(0)