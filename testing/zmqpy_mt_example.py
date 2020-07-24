"""
pyamzq server example

Binds REP socket to icp://trefide.ipc
Expects 6 block coordinates from client, sends blocks to server
"""
import os
import sys
import time
import threading
import zmq

import numpy as np


def worker(worker_url, context, mov):
    """Thread worker routine to recv requests and send chunks of a movie"""

    # Socket to talk to dispatcher
    socket = context.socket(zmq.REP)
    socket.connect(worker_url)

    while True:
        message = socket.recv()
        positions = np.frombuffer(message, dtype=np.intc)
        r_start, r_end, c_start, c_end, f_start, f_end = positions
        # print("Received request for: {} {} {} {} {} {}".format(r_start, r_end, c_start, c_end, f_start, f_end))

        if r_start == -1 or r_end == -1 or c_start == -1 or f_start == -1 or f_end == -1:
            return

        chunk = mov[r_start:r_end, c_start:c_end, f_start:f_end]
        # print("Sending data chunk:\n {}\n".format(chunk))
        socket.send(chunk.tobytes(order='F'))


def start_server(filepath):
    """Server routine"""
    url_worker = "inproc://workers"
    url_client = "ipc://trefide.ipc"

    # Prepare our context and sockets
    context = zmq.Context(io_threads=os.cpu_count())

    # Socket to talk to clients
    clients = context.socket(zmq.ROUTER)
    clients.bind(url_client)

    # Socket to talk to workers
    workers = context.socket(zmq.DEALER)
    workers.bind(url_worker)

    mov = np.load(filepath, mmap_mode='r')

    # Launch pool of worker threads
    threads = [threading.Thread(target=worker, args=(url_worker, context, mov)) for _ in range(os.cpu_count())]

    for thread in threads:
        thread.start()

    zmq.proxy(clients, workers)

    for thread in threads:
        thread.join()

    # We never get here but clean up anyhow
    clients.close()
    workers.close()
    context.term()


def main():
    """Starts the server when running this module"""
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        raise FileNotFoundError("{} is not a file".format(filepath))

    start_server(filepath)


if __name__ == "__main__":
    main()
