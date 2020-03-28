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
            GPUtil.showUtilization(all=True)
            print()
            print(f"CPU usage (mean %): {process.cpu_percent()}")
            print(f"System RAM usage (Gb): {process.memory_info()[0]/1e6}")
            print()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
