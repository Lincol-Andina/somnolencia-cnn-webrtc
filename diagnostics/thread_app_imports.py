import threading
import traceback


def target():
    try:
        import numpy
        import pandas
        from PIL import Image
        import streamlit as st
        import matplotlib.pyplot as plt
        import seaborn as sns
        import av
        import cv2
        import tensorflow as tf
        from tensorflow import keras
        import streamlit.components.v1 as components
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        print("exact app imports OK in thread", flush=True)
    except BaseException:
        traceback.print_exc()


worker = threading.Thread(target=target)
worker.start()
worker.join()
print("thread finished", flush=True)
