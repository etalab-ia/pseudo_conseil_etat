import GPUtil
from threading import Thread
import time
import psutil
import os
process = psutil.Process(os.getpid())

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            print()
            GPUtil.showUtilization(all=True)
            print(f"CPU usage (mean %): {process.cpu_percent()}")
            print(f"System RAM usage (Gb): {psutil._common.bytes2human(process.memory_info()[0])}")
            print()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
