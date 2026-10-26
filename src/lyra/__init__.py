"""Lyra (Lets You Run Anything): pipeline-parallel inference of HF models across machines on a tailnet.

wire.py      bytes on a socket: send/recv one tensor plus a JSON header
stage.py     one slice of the model: download only its weights, run its layers
autosplit.py measure every device and choose the fastest layer split
pipeline.py  the ring: workers (serve) and the driver (Driver)
tui.py       the terminal UI for the driver: devices, layer split, chat, tok/s
__main__.py  `lyra` opens the TUI; `lyra worker [port]` starts a worker
"""
