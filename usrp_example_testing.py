import numpy as np
import time
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
num_tx_samples = 6000         # number of samples to transmit
num_rx_samples = 6000         # number of samples to capture

# ---------------------------------------------------------------
# Digital communication system parameters.
# ---------------------------------------------------------------
fs = 1e6     # baseband sampling rate (1 MSps)
ts = 1 / fs  # sampling period (seconds per sample)
sps = 10     # samples per data symbol
T = ts * sps # symbol period (seconds per symbol)
t = np.arange(num_tx_samples) / fs # time vector

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


start = time.perf_counter()
# Schedule synchronized burst transmission and reception using hardware timestamps
tx_start = usrp.get_time_now() + 0.25      # schedule 1.0s in the future
rx_start = tx_start + 0.005

# Step 1: Send TX burst (queues to device FIFO and returns immediately)
signal_functions.stream_tx(tx_streamer, tx_metadata, tx_buffer, tx_start)

# Step 2: Issue timed RX command
signal_functions.stream_rx(rx_streamer, rx_cmd, rx_start)


# Step 3: Fetch captured RX samples (blocks until hardware receives them)
num_received = int(rx_streamer.recv(rx_buffer, rx_metadata, 5.0))
if num_received == 0:
    raise RuntimeError("No RX samples received.")
print(num_received)

# Close streamers
tx_streamer.close()                            # close transmit streamer
rx_streamer.close()                            # close receive streamer
end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")
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
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

# ax1.plot(f / 1e3, rx_fft_0, color="tab:blue")
# ax1.set_xlabel("Frequency (kHz)")
# ax1.set_ylabel("Magnitude")
# ax1.set_title("FFT of Received Signal - RX Channel 0")
# ax1.grid(True)

# ax2.plot(f / 1e3, rx_fft_1, color="tab:orange")
# ax2.set_xlabel("Frequency (kHz)")
# ax2.set_ylabel("Magnitude")
# ax2.set_title("FFT of Received Signal - RX Channel 1")
# ax2.grid(True)

# ---------------------------------------------------------------
# Create waterfall data (one FFT sample for each RX channel).
# ---------------------------------------------------------------
waterfall_0 = rx_fft_0[np.newaxis, :]
waterfall_1 = rx_fft_1[np.newaxis, :]

# ---------------------------------------------------------------
# Plot waterfall.
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 7), constrained_layout=True
)

extent = [f[0] / 1e3, f[-1] / 1e3, 0, 1]

im1 = ax1.imshow(
    waterfall_0,
    aspect="auto",
    origin="lower",
    extent=extent,
    cmap="viridis"
)
ax1.set_xlabel("Frequency (kHz)")
ax1.set_ylabel("Time (FFT sample)")
ax1.set_title("Waterfall - RX Channel 0")
fig.colorbar(im1, ax=ax1, label="Magnitude")

im2 = ax2.imshow(
    waterfall_1,
    aspect="auto",
    origin="lower",
    extent=extent,
    cmap="viridis"
)
ax2.set_xlabel("Frequency (kHz)")
ax2.set_ylabel("Time (FFT sample)")
ax2.set_title("Waterfall - RX Channel 1")
fig.colorbar(im2, ax=ax2, label="Magnitude")

plt.show()