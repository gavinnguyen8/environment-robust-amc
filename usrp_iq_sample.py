import os
import json
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import signal_functions

# ---------------------------------------------------------------
# Connect to USRP
# ---------------------------------------------------------------
from remoterf.drivers.usrp import MultiUSRP, uhd
usrp = MultiUSRP(token="USRP_00@ucla")

# ---------------------------------------------------------------
# USRP system parameters.
# ---------------------------------------------------------------
tx_carrier_freq_Hz = 915e6      # center RF frequency for TX (Hz), 70 MHz to 6 GHz
rx_carrier_freq_Hz = 915e6      # center RF frequency for RX (Hz), 70 MHz to 6 GHz
tx_gain_dB = 30                 # TX gain in dB (0 to ~89.8 dB)
rx_gain_dB = 30                 # RX gain in dB (0 to ~76 dB)
num_tx_samples = 7000         # number of samples to transmit
num_rx_samples = 7000         # number of samples to capture

# ---------------------------------------------------------------
# Digital communication system parameters.
# ---------------------------------------------------------------
fs = 1e6     # baseband sampling rate (1 MSps)
ts = 1 / fs  # sampling period (seconds per sample)
sps = 10     # samples per data symbol
T = ts * sps # symbol period (seconds per symbol)
t = np.arange(num_tx_samples) / fs # time vector

# ---------------------------------------------------------------
# IQ file export settings.
# ---------------------------------------------------------------
SAVE_NPY = True     # complex64 .npy per RX channel   -> np.load("file.npy")
SAVE_CFILE = True   # raw interleaved float32 I/Q .cfile per RX channel
                    # (GNU Radio File Source / inspectrum / np.fromfile)

# Save next to this script (falls back to the working directory if
# __file__ isn't defined, e.g. in a Jupyter notebook or REPL).
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# ---------------------------------------------------------------
# Create transmit signals (2 Channels).
# ---------------------------------------------------------------
# Channel 0: Complex exponential tone at +100 kHz
tone_freq_0 = 100e3
tx_signal_chan0 = signal_functions.create_tone(t, tone_freq_0)

# Channel 1: Complex exponential tone at +200 kHz
tone_freq_1 = 200e3
tx_signal_chan1 = signal_functions.create_tone(t, tone_freq_1)

# Scale signals to target digital amplitude (scaled by same factor)
tx_signal_chan0_scaled, tx_signal_chan1_scaled = signal_functions.scale(tx_signal_chan0, tx_signal_chan1)

# Set channels to be used
tx_channel = [0, 1]
rx_channel = [0, 1]

# Pack tx signals into buffer
tx_buffer = np.zeros((len(tx_channel), num_tx_samples), dtype=np.complex64)
tx_buffer[0] = tx_signal_chan0_scaled
tx_buffer[1] = tx_signal_chan1_scaled

# Create rx buffer for received signals
rx_buffer = np.zeros((len(rx_channel), num_rx_samples), dtype=np.complex64)

# Configure TX RF parameters across channel(s)
for ch in tx_channel:
    signal_functions.set_tx(usrp, ch, fs, tx_carrier_freq_Hz, tx_gain_dB)

# Configure RX RF parameters across channel(s)
for ch in rx_channel:
    signal_functions.set_rx(usrp, ch, fs, rx_carrier_freq_Hz, rx_gain_dB)

# Setup stream arguments: host fc32 (complex64), wire sc16 (16-bit integer)
tx_streamer, tx_metadata = signal_functions.create_tx_stream(usrp, tx_channel)
rx_streamer, rx_metadata, rx_cmd = signal_functions.create_rx_stream(usrp, rx_channel, num_rx_samples)

# Schedule synchronized burst transmission and reception using hardware timestamps
tx_start = usrp.get_time_now() + 1.0      # schedule 1.0s in the future
rx_start = tx_start + 0.005

# Step 1: Send TX burst (queues to device FIFO and returns immediately)
signal_functions.stream_tx(tx_streamer, tx_metadata, tx_buffer, tx_start)

# Step 2: Issue timed RX command
signal_functions.stream_rx(rx_streamer, rx_cmd, rx_start)


# Step 3: Fetch captured RX samples (blocks until hardware receives them)
num_received = signal_functions.fetch_rx(rx_streamer, rx_buffer, rx_metadata)

# Close streamers
tx_streamer.close()                            # close transmit streamer
rx_streamer.close()                            # close receive streamer

# ---------------------------------------------------------------
# Export received IQ samples (one file per RX channel + metadata).
# ---------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"iq_{rx_carrier_freq_Hz / 1e6:g}MHz_{fs / 1e6:g}Msps_{timestamp}"

saved_files = []
for idx, ch in enumerate(rx_channel):
    samples = np.ascontiguousarray(rx_buffer[idx, :num_received], dtype=np.complex64)
    stem = os.path.join(script_dir, f"{base_name}_ch{ch}")

    if SAVE_NPY:
        np.save(stem + ".npy", samples)
        saved_files.append(stem + ".npy")

    if SAVE_CFILE:
        samples.tofile(stem + ".cfile")      # interleaved float32: I0, Q0, I1, Q1, ...
        saved_files.append(stem + ".cfile")

# Sidecar metadata so the capture can be interpreted later
metadata = {
    "timestamp": datetime.now().isoformat(),
    "sample_rate_Hz": fs,
    "rx_center_freq_Hz": rx_carrier_freq_Hz,
    "rx_gain_dB": rx_gain_dB,
    "tx_center_freq_Hz": tx_carrier_freq_Hz,
    "tx_gain_dB": tx_gain_dB,
    "tx_tone_freqs_Hz": [tone_freq_0, tone_freq_1],
    "rx_channels": rx_channel,
    "num_samples_per_channel": int(num_received),
    "dtype": "complex64 (float32 I, float32 Q interleaved in .cfile)",
}
meta_path = os.path.join(script_dir, f"{base_name}_meta.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
saved_files.append(meta_path)

print("Saved IQ files:")
for path in saved_files:
    print(f"  {path}")

# ---------------------------------------------------------------
# Take FFT of received signals.
# ---------------------------------------------------------------
rx_chan0 = rx_buffer[0, :num_received]
rx_chan1 = rx_buffer[1, :num_received]

rx_fft_0 = np.abs(np.fft.fftshift(np.fft.fft(rx_chan0)))
rx_fft_1 = np.abs(np.fft.fftshift(np.fft.fft(rx_chan1)))
f = np.linspace(-fs / 2, fs / 2, num_received)

# ---------------------------------------------------------------
# Plot spectral response.
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

ax1.plot(f / 1e3, rx_fft_0, color="tab:blue")
ax1.set_xlabel("Frequency (kHz)")
ax1.set_ylabel("Magnitude")
ax1.set_title("FFT of Received Signal - RX Channel 0")
ax1.grid(True)

ax2.plot(f / 1e3, rx_fft_1, color="tab:orange")
ax2.set_xlabel("Frequency (kHz)")
ax2.set_ylabel("Magnitude")
ax2.set_title("FFT of Received Signal - RX Channel 1")
ax2.grid(True)

plt.show()