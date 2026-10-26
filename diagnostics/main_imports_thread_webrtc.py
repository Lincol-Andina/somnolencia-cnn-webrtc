import faulthandler
import threading
import traceback

faulthandler.enable()

import av
import cv2
import matplotlib.pyplot as plt
import numpy
import pandas
import pyarrow
import seaborn as sns
import streamlit
import tensorflow as tf
from PIL import Image

print("main imports OK", flush=True)


def target():
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        print("webrtc thread import OK", WebRtcMode, webrtc_streamer, flush=True)
    except BaseException:
        traceback.print_exc()


worker = threading.Thread(target=target)
worker.start()
worker.join()
print("thread finished", flush=True)
