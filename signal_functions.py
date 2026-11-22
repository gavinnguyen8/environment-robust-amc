import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Connect to USRP
# ---------------------------------------------------------------
from remoterf.drivers.usrp import MultiUSRP, uhd

def create_tone(t, tone_freq):
    '''
    Create a transmit signal tone
    '''
    signal_tone = np.exp(2.0j * np.pi * tone_freq * t).astype(np.complex64) # Create complex exponential tone
    return signal_tone

def scale(*signals):
    '''
    Scale a set number of signals to target digital amplitude (scaled by same factor)
    '''
    max_peak = max(np.max(np.abs(signal)) for signal in signals)  # grab max peak of collage tx signals
    target_amplitude = 1.0                      # target peak digital amplitude
    scale_factor = target_amplitude / max_peak  # uniform scale factor preserving relative amplitudes
    signals_scaled = tuple(signal * scale_factor for signal in signals)
    return signals_scaled

def set_tx(usrp, ch, fs, tx_carrier_freq_Hz, tx_gain_dB):
    '''
    Set transmitter parameters for specified channel
    '''
    usrp.set_tx_rate(fs, ch)          # set transmitter sampling rate
    usrp.set_tx_freq(tx_carrier_freq_Hz, ch)   # set transmitter carrier frequency
    usrp.set_tx_gain(tx_gain_dB, ch)           # set transmit gain

def set_rx(usrp, ch, fs, rx_carrier_freq_Hz, rx_gain_dB):
    '''
    Set receiver parameters for specified channel
    '''
    usrp.set_rx_rate(fs, ch)          # set transmitter sampling rate
    usrp.set_rx_freq(rx_carrier_freq_Hz, ch)   # set transmitter carrier frequency
    usrp.set_rx_gain(rx_gain_dB, ch)           # set transmit gain

def create_tx_stream(usrp, tx_channels):
    '''
    Create transmitter stream
    '''
    tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
    tx_args.channels = tx_channels
    tx_streamer = usrp.get_tx_stream(tx_args)

    tx_metadata = uhd.types.TXMetadata() # set information about transmitted data
    tx_metadata.start_of_burst = True              # marks start of burst
    tx_metadata.end_of_burst = True                # marks end of burst
    tx_metadata.has_time_spec = True               # enable hardware timestamping
    return tx_streamer, tx_metadata

def create_rx_stream(usrp, rx_channels, num_rx_samples):
    '''
    Create receiver stream
    '''
    rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_args.channels = rx_channels
    rx_streamer = usrp.get_rx_stream(rx_args)
    rx_metadata = uhd.types.RXMetadata() # stores information about the received data
    
    rx_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    rx_cmd.num_samps = num_rx_samples             # number of samples to receive
    rx_cmd.stream_now = False                     # do not stream immediately
    return rx_streamer, rx_metadata, rx_cmd

def stream_tx(tx_streamer, tx_metadata, tx_buffer, tx_start):
    '''
    Stream Tx Signal
    '''
    tx_metadata.time_spec = tx_start            # execute burst at timestamp
    tx_streamer.send(tx_buffer, tx_metadata, 5.0)  # submit transmit buffer, allows 5 seconds before reporting timeout

def stream_rx(rx_streamer, rx_cmd, rx_start):
    '''
    Stream Receiver
    '''
    rx_cmd.time_spec = rx_start        # begin capture 5 ms after TX burst start
    rx_streamer.issue_stream_cmd(rx_cmd)

def fetch_rx(rx_streamer, rx_buffer, rx_metadata):
    '''
    Fetch captured RX samples
    '''
    num_received = int(rx_streamer.recv(rx_buffer, rx_metadata, 5.0))
    if num_received == 0:
        raise RuntimeError("No RX samples received.")
    else:
        print("RX samples received:", num_received)
    return num_received
