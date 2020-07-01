"""
pyamzq server example

Binds REP socket to icp://trefide.ipc
Expects 6 block coordinates from client, sends blocks to server
"""
import os
import zmq

import numpy as np

if __name__ == "__main__":
    context = zmq.Context(io_threads=os.cpu_count())

    file_name = "large_dummy_movie.npy"
    mov = np.load(file_name, mmap_mode='r')

    address = "ipc://trefide.ipc"
    socket = context.socket(zmq.REP)
    socket.bind(address)
    # print("binded socket to address: {}".format(address))

    while True:
        message = socket.recv()
        positions = np.frombuffer(message, dtype=np.intc)
        r_start, r_end, c_start, c_end, f_start, f_end = positions
        # print("Received request for: {} {} {} {} {} {}".format(r_start, r_end, c_start, c_end, f_start, f_end))

        chunk = mov[r_start:r_end, c_start:c_end, f_start:f_end]
        # chunk = np.ascontiguousarray(chunk)
        # chunk = np.frombuffer(chunk)
        # print("Sending data chunk:\n {}\n".format(chunk))
        socket.send(chunk.tobytes(order='F'))
