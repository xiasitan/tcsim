import numpy as np




# use this if ecd char func is calibrated to sigma 1
#def func(xs, alpha, amp, ofs):
#    return amp * np.exp(-np.abs(xs)**2 / 2) * np.cos(2 * alpha * xs) + ofs

# use this if ecd char func is calibrated to sigma 0.5
def func(xs, alpha, amp, ofs):
    return amp * np.exp(-np.abs(xs*2)**2 / 2) * np.cos(2 * alpha * xs*2) + ofs


def guess(xs, ys):
    alpha = np.fft.rfftfreq(len(xs), xs[1] - xs[0])
    ofs = np.mean(ys)
    fft = np.fft.rfft(ys - ofs)
    idx = np.argmax(abs(fft))
    alpha0 = alpha[idx]
    amp = np.std(ys - ofs)
    # phi = np.angle(fft[idx])
    return {
        'alpha': (alpha0),
        'ofs': (ofs),
        # 'amp': (amp, 0, np.max(ys) - np.min(ys)),
        'amp': (amp),
        # 'phi': (phi, -2*np.pi, 2*np.pi),
    }
