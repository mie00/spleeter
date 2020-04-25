#!/usr/bin/env python
# coding: utf8

"""
    Entrypoint provider for tcp server.

    USAGE: python server.py
"""

import socketserver
import signal
from .separator import Separator

import numpy as np

__email__ = 'mohamed@elawadi.net'
__author__ = 'Mohamed Elawadi'
__license__ = 'MIT License'

should_process = True

def handler(signum, frame):
    global should_process
    should_process = not should_process
    if should_process:
        print("converting")
    else:
        print("not converting")


signal.signal(signal.SIGUSR1, handler)

sep = Separator(
    'spleeter:2stem-finetune-realtime',
    MWF=False,
    stft_backend='tensorflow',
    multiprocess=False)

class Handler_TCPServer(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            longdt = np.dtype('int64').newbyteorder('<')
            floatdt = np.dtype('float32').newbyteorder('<')
            sample_rate = np.frombuffer(self.request.recv(8), dtype=longdt)[0]
            size = np.frombuffer(self.request.recv(8), dtype=longdt)[0]
            received = self.request.recv(size * 4)
            while len(received) < size * 4:
                received += self.request.recv(size * 4 - len(received))
            ret = received
            if should_process:
                data = np.frombuffer(received, dtype=floatdt).astype(
                    'float64').reshape((-1, 1))
                c = sep.separate(data)
                ret = c['vocals'][:, 1].astype('float32').tobytes()
            self.request.sendall(ret)


HOST, PORT = "localhost", 8083
tcp_server = socketserver.ThreadingTCPServer((HOST, PORT), Handler_TCPServer)
print("warming up")
sep.separate(np.zeros((1024, 2)))
print("serving on :8083")
tcp_server.serve_forever()
