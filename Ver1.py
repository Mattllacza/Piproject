# ================================================
# CLAP DATA COLLECTION VERSION (NO SIGNATURE FILTER)
# For ReSpeaker USB Mic Array (6 channels)
# ================================================

import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
from collections import deque
import time

# ---------------- PARAMETERS ----------------
SR = 16000
FRAME = 1024
DEVICE = 1           # <<< SET THIS TO YOUR RESPEAKER INDEX
CHANNELS = 6         # ReSpeaker = 6 channels
LOW_HZ = 1200.0
HIGH_HZ = 4500.0
FILTER_ORDER = 4

ADAPT_BUF = 120
ATTACK_BUF = 6

# Relative thresholds for coarse detection (keep)
TUNE_FFT_MULT = 3.0
TUNE_RMS_MULT = 3.0
TUNE_ATTACK_MULT = 2.5
MIN_BAND_RATIO = 0.6

# Decay-based confirmation
DECAY_FRAMES = 2
DECAY_ALPHA = 0.6
DECAY_BETA = 0.4
COOLDOWN = 0.6      # seconds

# ---------------- SETUP ----------------
def make_bandpass(low, high, fs, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return b, a

b, a = make_bandpass(LOW_HZ, HIGH_HZ, SR, order=FILTER_ORDER)
WIN = np.hanning(FRAME)
freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
band_idx = np.where((freqs >= LOW_HZ) & (freqs <= HIGH_HZ))[0]

fft_median_buf = deque(maxlen=ADAPT_BUF)
rms_median_buf = deque(maxlen=ADAPT_BUF)
recent_band_buf = deque(maxlen=ATTACK_BUF)

last_clap_time = 0.0

pending_candidate = None
pending_future_energies = []
pending_frames_left = 0

# ------------- FEATURE EXTRACTION -------------
def block_band_energy(block):
    X = np.fft.rfft(block * WIN)
    power = np.abs(X)**2
    band = float(power[band_idx].sum())
    total = float(power.sum()) + 1e-12
    return band, total

def compute_features(block):
    xf = lfilter(b, a, block)
    rms = float(np.sqrt(np.mean(xf*xf) + 1e-12))
    peak = float(np.max(np.abs(block)))
    band_energy, total_energy = block_band_energy(block)
    band_ratio = band_energy / (total_energy + 1e-12)
    attack_slope = float(np.max(np.diff(np.abs(block))))
    duration = len(block) / SR
    return {
        "rms": rms,
        "peak": peak,
        "band_energy": band_energy,
        "band_ratio": band_ratio,
        "attack_slope": attack_slope,
        "duration": duration,
    }

# ------------- DECAY CHECK ----------------
def passes_decay(candidate_energy, future_energies):
    if not future_energies:
        return False
    first = future_energies[0]
    min_future = min(future_energies)
    r1 = first / (candidate_energy + 1e-12)
    rmin = min_future / (candidate_energy + 1e-12)
    return (r1 < DECAY_ALPHA) and (rmin < DECAY_BETA)

# ------------- CALLBACK ----------------
def on_audio(indata, frames, time_info, status):
    global last_clap_time
    global pending_candidate, pending_future_energies, pending_frames_left

    if status:
        print("Audio status:", status)

    # Use beamformed channel 0 for detection
    block = indata[:, 0].astype(np.float32)
    feat = compute_features(block)

    # Update background
    fft_median_buf.append(feat["band_energy"])
    rms_median_buf.append(feat["rms"])
    recent_band_buf.append(feat["band_energy"])

    # Background calibration
    if len(fft_median_buf) < 8:
        print(f"Calibrating... {len(fft_median_buf)}/{fft_median_buf.maxlen}", end="\r")
        return

    median_fft = float(np.median(fft_median_buf))
    median_rms = float(np.median(rms_median_buf))
    median_recent = float(np.median(list(recent_band_buf)[:-1] or [median_fft]))

    now = time.time()

    # ---------- Pending candidate check (DECAY) ----------
    if pending_candidate is not None:
        pending_future_energies.append(feat["band_energy"])
        pending_frames_left -= 1

        if pending_frames_left <= 0:

            cand = pending_candidate
            energy_cand = cand["band_energy"]
            ok_decay = passes_decay(energy_cand, pending_future_energies)

            if ok_decay and (now - last_clap_time) > COOLDOWN:
                last_clap_time = now
                print("\n" + "="*70)
                print(f"[{time.strftime('%H:%M:%S')}] CLAP CONFIRMED!")
                print("Candidate features:")
                for k,v in cand.items():
                    print(f"  {k:12s} = {v}")
                print("="*70 + "\n")
            else:
                print("\n" + "-"*70)
                print(f"[{time.strftime('%H:%M:%S')}] Candidate rejected (decay).")
                print("Candidate features:")
                for k,v in cand.items():
                    print(f"  {k:12s} = {v}")
                print("-"*70 + "\n")

            # Reset candidate
            pending_candidate = None
            pending_future_energies = []
            pending_frames_left = 0

        return

    # ---------- COARSE DETECTION ----------
    cond_fft    = feat["band_energy"] > median_fft * TUNE_FFT_MULT
    cond_rms    = feat["rms"]         > median_rms * TUNE_RMS_MULT
    cond_attack = feat["band_energy"] > median_recent * TUNE_ATTACK_MULT
    cond_ratio  = feat["band_ratio"]  > MIN_BAND_RATIO

    if cond_fft and cond_rms and cond_attack and cond_ratio:
        pending_candidate = feat
        pending_future_energies = []
        pending_frames_left = DECAY_FRAMES

        print("\n" + "-"*70)
        print(f"[{time.strftime('%H:%M:%S')}] Coarse candidate detected")
        print("Features:")
        for k,v in feat.items():
            print(f"  {k:12s} = {v}")
        print("-"*70)
        return

    # Debug line (when idle)
    print(
        f"raw_peak={feat['peak']:.4f}  "
        f"rms={feat['rms']:.6f}  "
        f"band_ratio={feat['band_ratio']:.3f}",
        end="\r"
    )

# ------------- MAIN ----------------
def main():
    print("Using device:", sd.query_devices(DEVICE)['name'])
    print("Starting clap data collector (signature disabled).")
    try:
        with sd.InputStream(device=DEVICE,
                            channels=CHANNELS,
                            samplerate=SR,
                            blocksize=FRAME,
                            callback=on_audio,
                            dtype="float32"):
            print("Listening... Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print("Stream error:", e)

if __name__ == "__main__":
    main()
