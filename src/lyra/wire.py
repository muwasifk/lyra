"""
Sending tensors between machines: one message = a JSON header plus (optionally) one tensor's raw bytes.

On the wire: [4 bytes: header length][4 bytes: payload length][JSON header][payload]
"""

import json
import socket
import struct

import torch


def send_msg(sock, meta, t=None):
    payload = b""
    if t is not None:
        t = t.detach().contiguous().cpu()
        meta = {**meta, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
        payload = t.view(torch.uint8).numpy().tobytes()  # numpy has no bf16, so ship raw bytes
    header = json.dumps(meta).encode()
    sock.sendall(struct.pack("!II", len(header), len(payload)) + header + payload)


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:  # recv returns whatever has arrived so far, not necessarily n bytes
        k = sock.recv_into(view[got:])
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
    return buf


def recv_msg(sock, device):
    hlen, plen = struct.unpack("!II", recv_exact(sock, 8))
    meta = json.loads(recv_exact(sock, hlen))
    t = None
    if plen:
        raw = torch.frombuffer(recv_exact(sock, plen), dtype=torch.uint8)
        t = raw.view(getattr(torch, meta["dtype"])).reshape(meta["shape"]).to(device)
    return meta, t


def connect(host, port):
    sock = socket.create_connection((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # send small messages now, don't batch them
    return sock


def accept(srv):
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn
