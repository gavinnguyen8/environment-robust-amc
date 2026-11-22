"""
Plot IQ files exported by tx_rx_tones_iq_export.py.

Usage:
    python plot_iq.py                      # plots the most recent capture in this folder
    python plot_iq.py FILE [FILE ...]      # plots specific .npy / .cfile files
    python plot_iq.py --fs 2e6 --fc 2.4e9 FILE   # override sample rate / center freq
    python plot_iq.py --save               # also save a PNG next to each IQ file

Sample rate and center frequency are read from the matching *_meta.json
when it exists next to the IQ file; --fs / --fc override it.
"""
import os
import re
import sys
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Plot settings.
# ---------------------------------------------------------------
TIME_PLOT_SAMPLES = 200   # number of samples shown in the time-domain plot
SPEC_NFFT = 256           # FFT size for the spectrogram (waterfall)
DEFAULT_FS = 1e6          # used only if no metadata file and no --fs

# Folder this script lives in (falls back to the working directory
# if __file__ isn't defined, e.g. in a REPL).
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()


# ---------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------
def resolve_path(p):
    """Accept a full path, a path relative to the cwd, or a bare file name in the script folder."""
    if os.path.isfile(p):
        return os.path.abspath(p)
    alt = os.path.join(script_dir, p)
    if os.path.isfile(alt):
        return alt
    sys.exit(f"File not found: {p}")


def load_iq(path):
    """Load complex IQ samples from .npy or raw interleaved float32 (.cfile/.bin/.dat/.iq)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        x = np.load(path)
    elif ext in (".cfile", ".bin", ".dat", ".iq"):
        x = np.fromfile(path, dtype=np.complex64)
    else:
        sys.exit(f"Unsupported file type '{ext}' for {path} (use .npy or .cfile)")
    x = np.asarray(x, dtype=np.complex64)
    if x.ndim != 1:
        sys.exit(f"Expected a 1-D IQ array in {path}, got shape {x.shape}")
    return x


def load_meta(path):
    """Load the *_meta.json that goes with an IQ file (returns {} if there isn't one)."""
    base = re.sub(r"_ch\d+\.[^.]+$", "", os.path.basename(path))
    meta_path = os.path.join(os.path.dirname(path), base + "_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def latest_capture_files():
    """Return the per-channel IQ files of the most recent capture in the script folder."""
    metas = glob.glob(os.path.join(script_dir, "iq_*_meta.json"))
    if not metas:
        sys.exit(f"No iq_*_meta.json found in {script_dir}. "
                 "Run the capture script first or pass IQ file(s) explicitly.")
    base = max(metas, key=os.path.getmtime)[:-len("_meta.json")]
    files = sorted(glob.glob(base + "_ch*.npy")) or sorted(glob.glob(base + "_ch*.cfile"))
    if not files:
        sys.exit(f"Found {base}_meta.json but no matching _ch*.npy / _ch*.cfile files.")
    return files


# ---------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------
def plot_iq(x, fs, fc, title):
    n = len(x)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=11)

    # --- Time domain (I and Q) ---
    ax = axs[0, 0]
    n_t = min(n, TIME_PLOT_SAMPLES)
    t_us = np.arange(n_t) / fs * 1e6
    ax.plot(t_us, x[:n_t].real, label="I", color="tab:blue")
    ax.plot(t_us, x[:n_t].imag, label="Q", color="tab:orange")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Time domain (first {n_t} samples)")
    ax.margins(y=0.2)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True)

    # --- Spectrum (Hann-windowed FFT magnitude, dB) ---
    ax = axs[0, 1]
    w = np.hanning(n)
    X = np.fft.fftshift(np.fft.fft(x * w))
    mag_db = 20 * np.log10(np.abs(X) / np.sum(w) + 1e-12)
    f_khz = np.fft.fftshift(np.fft.fftfreq(n, d=1 / fs)) / 1e3
    ax.plot(f_khz, mag_db, color="tab:green")
    ax.set_xlabel("Frequency offset (kHz)" if fc is None else f"Offset from {fc / 1e6:g} MHz (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Spectrum")
    ax.grid(True)

    # --- Spectrogram / waterfall ---
    # Fs is passed in kHz so the frequency axis comes out in kHz and the time axis in ms.
    ax = axs[1, 0]
    nfft = min(SPEC_NFFT, n)
    im = ax.specgram(x, NFFT=nfft, Fs=fs / 1e3, noverlap=nfft // 2, cmap="viridis")[3]
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency offset (kHz)")
    ax.set_title("Spectrogram")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    # --- Constellation (I vs Q) ---
    ax = axs[1, 1]
    ax.scatter(x.real, x.imag, s=2, alpha=0.4, color="tab:purple")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("IQ scatter")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    return fig


# ---------------------------------------------------------------
# Main.
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot exported IQ files.")
    parser.add_argument("files", nargs="*", help="IQ file(s) (.npy / .cfile). Default: latest capture in this folder.")
    parser.add_argument("--fs", type=float, default=None, help="sample rate in Hz (overrides metadata)")
    parser.add_argument("--fc", type=float, default=None, help="center frequency in Hz (overrides metadata)")
    parser.add_argument("--save", action="store_true", help="save a PNG next to each IQ file")
    args = parser.parse_args()

    paths = [resolve_path(p) for p in args.files] if args.files else latest_capture_files()

    for path in paths:
        x = load_iq(path)
        if len(x) == 0:
            print(f"Skipping empty file: {path}")
            continue

        meta = load_meta(path)
        fs = args.fs or meta.get("sample_rate_Hz")
        fc = args.fc or meta.get("rx_center_freq_Hz")
        if fs is None:
            fs = DEFAULT_FS
            print(f"No sample rate in metadata or --fs; assuming {fs / 1e6:g} MS/s")

        print(f"{os.path.basename(path)}: {len(x)} samples, fs = {fs / 1e6:g} MS/s, "
              f"duration = {len(x) / fs * 1e3:.2f} ms")

        title = f"{os.path.basename(path)}   |   N = {len(x)},  fs = {fs / 1e6:g} MS/s"
        if fc is not None:
            title += f",  fc = {fc / 1e6:g} MHz"
        fig = plot_iq(x, fs, fc, title)

        if args.save:
            png_path = os.path.splitext(path)[0] + "_plot.png"
            fig.savefig(png_path, dpi=150)
            print(f"  saved {png_path}")

    plt.show()


if __name__ == "__main__":
    main()
