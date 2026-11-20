import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Connect to USRP
# ---------------------------------------------------------------
from remoterf.drivers.usrp import MultiUSRP, uhd
usrp = MultiUSRP(token="USRP_00@ucla") 

# ---------------------------------------------------------------
# Digital communication system parameters.
# ---------------------------------------------------------------
fs = 1e6     # baseband sampling rate (1 MSps)
ts = 1 / fs  # sampling period (seconds per sample)
sps = 10     # samples per data symbol
T = ts * sps # symbol period (seconds per symbol)

# ---------------------------------------------------------------
# USRP system parameters.
# ---------------------------------------------------------------
sample_rate = fs                # baseband sampling rate (Hz), up to 56 MSps
tx_carrier_freq_Hz = 915e6      # center RF frequency for TX (Hz), 70 MHz to 6 GHz
rx_carrier_freq_Hz = 915e6      # center RF frequency for RX (Hz), 70 MHz to 6 GHz
tx_gain_dB = 30                 # TX gain in dB (0 to ~89.8 dB)
rx_gain_dB = 30                 # RX gain in dB (0 to ~76 dB)
num_tx_samples = 7_000         # number of samples to transmit
num_rx_samples = 7_000         # number of samples to capture

# ---------------------------------------------------------------
# Create transmit signals (2 Channels).
# ---------------------------------------------------------------
t = np.arange(num_tx_samples) / sample_rate # time vector

# Channel 0: Complex exponential tone at +100 kHz
tone_freq_0 = 100e3                         # tone offset frequency for channel 0
tx_signal_chan0 = np.exp(2.0j * np.pi * tone_freq_0 * t).astype(np.complex64)
# Channel 1: Complex exponential tone at +200 kHz
tone_freq_1 = 200e3                         # tone offset frequency for channel 1
tx_signal_chan1 = np.exp(2.0j * np.pi * tone_freq_1 * t).astype(np.complex64)




# Scale signals to target digital amplitude (scaled by same factor)
target_amplitude = 1.0                      # target peak digital amplitude
max_peak = max(np.max(np.abs(tx_signal_chan0)), np.max(np.abs(tx_signal_chan1)))
scale_factor = target_amplitude / max_peak  # uniform scale factor preserving relative amplitudes



tx_signal_chan0_scaled = tx_signal_chan0 * scale_factor
tx_signal_chan1_scaled = tx_signal_chan1 * scale_factor

# Pack scaled signals into 2D buffer (shape: [num_channels, num_samples])
tx_buffer = np.zeros((2, num_tx_samples), dtype=np.complex64)
tx_buffer[0] = tx_signal_chan0_scaled
tx_buffer[1] = tx_signal_chan1_scaled




# Configure RF parameters across both channels (0 and 1)
for ch in [0, 1]:
    usrp.set_tx_rate(sample_rate, ch)          # set transmitter sampling rate
    usrp.set_tx_freq(tx_carrier_freq_Hz, ch)   # set transmitter carrier frequency
    usrp.set_tx_gain(tx_gain_dB, ch)           # set transmit gain

    usrp.set_rx_rate(sample_rate, ch)          # set receiver sampling rate
    usrp.set_rx_freq(rx_carrier_freq_Hz, ch)   # set receiver carrier frequency
    usrp.set_rx_gain(rx_gain_dB, ch)           # set receive gain

# Setup stream arguments: host fc32 (complex64), wire sc16 (16-bit integer)
rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_args.channels = [0, 1]
rx_streamer = usrp.get_rx_stream(rx_args)
rx_buffer = np.zeros((2, num_rx_samples), dtype=np.complex64)
rx_metadata = uhd.types.RXMetadata()
tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
tx_args.channels = [0, 1]
tx_streamer = usrp.get_tx_stream(tx_args)
# Schedule synchronized burst transmission and reception using hardware timestamps
burst_start = usrp.get_time_now() + 1.0        # schedule 1.0s in the future

# Step 1: Issue timed RX command
rx_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
rx_cmd.num_samps = num_rx_samples             # number of samples to receive
rx_cmd.stream_now = False                     # do not stream immediately
rx_cmd.time_spec = burst_start + 0.005        # begin capture 5 ms after TX burst start
rx_streamer.issue_stream_cmd(rx_cmd)

# Step 2: Send TX burst (queues to device FIFO and returns immediately)
tx_metadata = uhd.types.TXMetadata()
tx_metadata.start_of_burst = True              # marks start of burst
tx_metadata.end_of_burst = True                # marks end of burst
tx_metadata.has_time_spec = True               # enable hardware timestamping
tx_metadata.time_spec = burst_start            # execute burst at timestamp
tx_streamer.send(tx_buffer, tx_metadata, 5.0)  # submit transmit buffer
# Step 3: Fetch captured RX samples (blocks until hardware receives them)
num_received = int(rx_streamer.recv(rx_buffer, rx_metadata, 5.0))
if num_received == 0:
    raise RuntimeError("No RX samples received.")

# Close streamers
tx_streamer.close()                            # close transmit streamer
rx_streamer.close()                            # close receive streamer

# ---------------------------------------------------------------
# Take FFT of received signals.
# ---------------------------------------------------------------
rx_chan0 = rx_buffer[0, :num_received]
rx_chan1 = rx_buffer[1, :num_received]

rx_fft_0 = np.abs(np.fft.fftshift(np.fft.fft(rx_chan0)))
rx_fft_1 = np.abs(np.fft.fftshift(np.fft.fft(rx_chan1)))
f = np.linspace(-sample_rate / 2, sample_rate / 2, num_received)

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