"""Entry point.

uv run lyra                                the terminal UI: this machine is the driver
uv run lyra worker [port] [--memory GB]    a worker: waits for a driver to send it a plan, then runs its layers
"""

import argparse
import os
import sys

from .pipeline import PORT


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        from .pipeline import serve
        from .stage import default_device

        parser = argparse.ArgumentParser(prog="lyra worker")
        parser.add_argument("port", nargs="?", type=int, default=PORT, help=f"port to listen on (default {PORT})")
        parser.add_argument(
            "--memory",
            type=float,
            metavar="GB",
            help="the most GPU memory (or, on a Mac, unified memory) to offer; default: all that's free",
        )
        args = parser.parse_args(sys.argv[2:])

        device = os.environ.get("LYRA_DEVICE") or default_device()
        offered_bytes = None
        if args.memory is not None:
            offered_bytes = int(args.memory * 1e9)
        serve(args.port, device, offered_bytes)
    else:
        from .tui import main as tui_main  # only the driver needs textual, so import it only here

        tui_main()


if __name__ == "__main__":
    main()
