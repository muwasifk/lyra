"""Send tensors to another machine and back, check they arrive intact, and time the round trip.

On the receiving machine:  uv run python scripts/tensor_ping.py serve
On the sending machine:    uv run python scripts/tensor_ping.py send <hostname>
"""

import argparse
import socket
import time

import torch

from lyra.stage import default_device
from lyra.wire import accept, connect, recv_msg, send_msg

PORT = 7070

# Hidden-state sizes for Qwen3-8B (hidden size 4096): one token while decoding, a 512-token prompt while prefilling.
SHAPES = {"decode": (1, 1, 4096), "prefill": (1, 512, 4096)}


def serve():
    """Echo every tensor straight back to whoever sent it. Never exits."""
    device = default_device()
    hostname = socket.gethostname()
    srv = socket.create_server(("", PORT))
    print(f"[{hostname}] listening on :{PORT} ({device})")

    while True:
        conn = accept(srv)
        print(f"connection from {conn.getpeername()[0]}")
        try:
            while True:
                message, tensor = recv_msg(conn, device)
                message["from"] = hostname  # so the sender can see who echoed it
                message["device"] = device
                send_msg(conn, message, tensor)
        except ConnectionError:
            print("peer disconnected")
        finally:
            conn.close()


def send(host, rounds):
    """Send each tensor size `rounds` times, check it comes back unchanged, and print the round-trip times."""
    device = default_device()
    sock = connect(host, PORT)

    for name, shape in SHAPES.items():
        sent = torch.randn(shape, dtype=torch.bfloat16, device=device)
        times_ms = []
        for i in range(rounds):
            start_time = time.perf_counter()
            send_msg(sock, {"round": i}, sent)
            message, received = recv_msg(sock, device)
            times_ms.append((time.perf_counter() - start_time) * 1000)

            assert torch.equal(sent, received), "tensor came back different"
            assert message["round"] == i, "replies came back out of order"

        times_ms.sort()
        median = times_ms[len(times_ms) // 2]
        size_kb = sent.numel() * sent.element_size() / 1024
        print(
            f"{name:8s} {size_kb:7.0f} KB  round trip median {median:6.2f} ms  "
            f"(min {times_ms[0]:.2f}, max {times_ms[-1]:.2f})  via {message['from']} on {message['device']}"
        )

    sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve")
    send_parser = commands.add_parser("send")
    send_parser.add_argument("host", help="tailnet hostname or IP of the machine running `serve`")
    send_parser.add_argument("-n", type=int, default=50, help="round trips per tensor size")
    args = parser.parse_args()

    if args.command == "serve":
        serve()
    else:
        send(args.host, args.n)
