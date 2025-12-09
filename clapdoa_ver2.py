#!/usr/bin/env python3
import time
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import usb.core
import os
import sys

sys.path.append(os.path.expanduser('~/usb_4_mic_array'))
from tuning import Tuning

# config
SR = 16000
FRAME = 1024
DEVICE = 1
CHANNELS = 6
LOW_HZ = 1200.0
HIGH_HZ = 4500.0
FILTER_ORDER = 4
ADAPT_BUF = 120
ATTACK_BUF = 6
TUNE_FFT_MULT = 3.0
TUNE_RMS_MULT = 3.0
TUNE_ATTACK_MULT = 2.5
MIN_BAND_RATIO = 0.6
DECAY_FRAMES = 2
DECAY_ALPHA = 0.6
DECAY_BETA = 0.4
COOLDOWN = 0.6

# state
b = None
a = None
WIN = None
band_idx = None
fft_median_buf = deque(maxlen=ADAPT_BUF)
rms_median_buf = deque(maxlen=ADAPT_BUF)
recent_band_buf = deque(maxlen=ATTACK_BUF)

last_clap_time = 0.0
pending_candidate = None
pending_future_energies = []
pending_frames_left = 0
tuning = None

def make_bandpass(low, high, fs, order=4):
    return butter(order, [low/(fs/2), high/(fs/2)], btype='band')

def init_filters():
    global b, a, WIN, band_idx
    b, a = make_bandpass(LOW_HZ, HIGH_HZ, SR, FILTER_ORDER)
    WIN = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1.0/SR)
    band_idx = np.where((freqs >= LOW_HZ) & (freqs <= HIGH_HZ))[0]

def init_doa():
    global tuning
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if dev is None:
        raise RuntimeError("ReSpeaker not found.")
    tuning = Tuning(dev)
    print("DOA ready:", tuning.direction)

def block_band_energy(block):
    X = np.fft.rfft(block * WIN)
    p = np.abs(X)**2
    return float(p[band_idx].sum()), float(p.sum()) + 1e-12

def compute_features(block):
    xf = lfilter(b, a, block)
    rms = float(np.sqrt(np.mean(xf*xf) + 1e-12))
    peak = float(np.max(np.abs(block)))
    band_e, total_e = block_band_energy(block)
    ratio = band_e / (total_e + 1e-12)
    attack = float(np.max(np.diff(np.abs(block))))
    return {
        "rms": rms,
        "peak": peak,
        "band_energy": band_e,
        "band_ratio": ratio,
        "attack_slope": attack
    }

def passes_decay(e_cand, future):
    if not future:
        return False
    r1 = future[0] / (e_cand + 1e-12)
    rmin = min(future) / (e_cand + 1e-12)
    return (r1 < DECAY_ALPHA) and (rmin < DECAY_BETA)

def on_audio(indata, frames, time_info, status):
    global last_clap_time
    global pending_candidate, pending_future_energies, pending_frames_left

    block = indata[:, 0].astype(np.float32)
    feat = compute_features(block)

    fft_median_buf.append(feat["band_energy"])
    rms_median_buf.append(feat["rms"])
    recent_band_buf.append(feat["band_energy"])

    if len(fft_median_buf) < 8:
        return

    median_fft = float(np.median(fft_median_buf))
    median_rms = float(np.median(rms_median_buf))
    median_recent = float(np.median(list(recent_band_buf)[:-1] or [median_fft]))
    now = time.time()

    # decay phase
    if pending_candidate is not None:
        pending_future_energies.append(feat["band_energy"])
        pending_frames_left -= 1

        if pending_frames_left <= 0:
            cand = pending_candidate
            ok = passes_decay(cand["band_energy"], pending_future_energies)

            if ok and (now - last_clap_time) > COOLDOWN:
                last_clap_time = now
                print("CLAP CONFIRMED")

                angle = None
                if tuning is not None:
                    try: angle = tuning.direction
                    except: angle = None

                if angle is not None:
                    print("DOA:", angle)
                else:
                    print("DOA unavailable")

            pending_candidate = None
            pending_future_energies = []
            pending_frames_left = 0
        return

    # coarse detection
    cond_fft = feat["band_energy"] > median_fft * TUNE_FFT_MULT
    cond_rms = feat["rms"] > median_rms * TUNE_RMS_MULT
    cond_attack = feat["band_energy"] > median_recent * TUNE_ATTACK_MULT
    cond_ratio = feat["band_ratio"] > MIN_BAND_RATIO

    if cond_fft and cond_rms and cond_attack and cond_ratio:
        pending_candidate = feat
        pending_future_energies = []
        pending_frames_left = DECAY_FRAMES
        print("Candidate detected")
        return

def main():
    init_filters()
    init_doa()
    print("Device:", sd.query_devices(DEVICE)['name'])
    print("Listening...")

    try:
        with sd.InputStream(
            device=DEVICE,
            channels=CHANNELS,
            samplerate=SR,
            blocksize=FRAME,
            callback=on_audio,
            dtype="float32"
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
