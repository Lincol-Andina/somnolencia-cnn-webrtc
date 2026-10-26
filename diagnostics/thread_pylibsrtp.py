import threading
import traceback


def target():
    try:
        import pylibsrtp

        print("pylibsrtp OK in thread", pylibsrtp.__version__, flush=True)
    except BaseException:
        traceback.print_exc()


worker = threading.Thread(target=target)
worker.start()
worker.join()
print("thread finished", flush=True)
