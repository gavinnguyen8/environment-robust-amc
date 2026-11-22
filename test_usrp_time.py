import numpy as np
import time
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Connect to USRP
# ---------------------------------------------------------------
from remoterf.drivers.usrp import MultiUSRP, uhd


usrp = MultiUSRP(token="USRP_00@ucla")

for i in range(10):
    tx_start = usrp.get_time_now() + 1.0      # schedule 1.0s in the future
    real_time = tx_start.get_real_secs()
    print(real_time)
    time.sleep(0.5)

