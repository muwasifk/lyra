"""Terminal UI for the driver: set up devices and the layer split, chat, and watch tok/s.

uv run lyra
"""

import gc
import json
import os
import socket
import subprocess
import time

import torch
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalGroup, Vertical, VerticalGroup, VerticalScroll
from textual.message import Message
from textual.suggester import SuggestFromList
from textual.widgets import Button, Footer, Input, Label, Markdown, RichLog, Static, Switch

from .pipeline import Driver, parse_address, parse_plan
from .stage import default_device

DEFAULT_MODEL = "Qwen/Qwen3-8B"
STATUS = {  # state -> (text, colour) for the status in the top bar
    "idle": ("not connected", "$text-muted"),
    "measuring": ("measuring devices…", "$warning"),
    "connecting": ("connecting…", "$warning"),
    "ready": ("ready", "$success"),
    "busy": ("generating…", "$accent"),
    "error": ("error · see log", "$error"),
}


def tailnet_hosts():
    """(this machine's tailnet name, names of the other machines that are online), from `tailscale status`."""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"], capture_output=True, text=True, timeout=3, check=False
        )
        status = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return socket.gethostname(), []
    me = status["Self"]["DNSName"].split(".")[0]  # MagicDNS name, e.g. "fedora"
    others = []
    for peer in (status.get("Peer") or {}).values():
        if peer.get("Online"):
            others.append(peer["DNSName"].split(".")[0])
    return me, sorted(others)


def split_reply(text):
    """(thinking, answer) of a reply so far. Models with a thinking mode start with <think> ... </think>."""
    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return "", text
    thought, _closed, answer = stripped[len("<think>") :].partition("</think>")
    return thought.strip("\n"), answer.lstrip("\n")


# Messages from the background threads to the UI. post_message is safe to call from any thread.
class RingReady(Message):
    def __init__(self, driver):
        super().__init__()
        self.driver = driver


class RingFailed(Message):
    def __init__(self, error):
        super().__init__()
        self.error = error


class TokenArrived(Message):
    def __init__(self, text, when):
        super().__init__()
        self.text = text
        self.when = when


class ReplyDone(Message):
    def __init__(self, reply, stats):
        super().__init__()
        self.reply = reply
        self.stats = stats


class AutoSplitDone(Message):
    def __init__(self, counts):
        super().__init__()
        self.counts = counts  # one layer count per device row, in order


class AutoSplitFailed(Message):
    def __init__(self, error):
        super().__init__()
        self.error = error


class ModelInfo(Message):
    def __init__(self, model, layers, error=None):
        super().__init__()
        self.model = model
        self.layers = layers
        self.error = error


class DeviceRow(HorizontalGroup):
    """One machine in the plan: its tailnet name (optionally :port) and how many layers it runs."""

    def __init__(self, host="", layers="", driver=False, hosts=()):
        super().__init__()
        self.start_host = host
        self.start_layers = layers
        self.is_driver = driver
        self.hosts = list(hosts)

    def compose(self) -> ComposeResult:
        yield Label("driver" if self.is_driver else "worker", classes="role")
        yield Input(
            self.start_host,
            placeholder="host[:port]",
            classes="host",
            suggester=SuggestFromList(self.hosts),
            compact=True,
        )
        yield Input(self.start_layers, placeholder="—", classes="layers", type="integer", compact=True)
        if self.is_driver:
            yield Label("", classes="spacer")  # lines the columns up with the worker rows' × button
        else:
            yield Button("×", classes="remove", compact=True)

    @on(Button.Pressed, ".remove")
    def remove_row(self):
        self.remove()
        self.app.call_after_refresh(self.app.refresh_plan)

    def read(self):
        host = self.query_one(".host", Input).value.strip()
        layers = self.query_one(".layers", Input).value.strip()
        return host, int(layers) if layers.isdigit() else 0


class EventLog(RichLog):
    """Receives everything the driver code prints (loading progress, "ring up", warnings)."""

    def on_print(self, event: events.Print) -> None:
        if event.text.strip():
            self.write(event.text.rstrip())


class ChatEntry(Vertical):
    """
    One chat message: a small label saying who wrote it ("you", or the model's name), then the text.

    A reply grows as tokens arrive: its thinking goes into a dimmed block of plain text, and its answer is
    rendered as markdown (headings, lists, tables, code blocks...).
    """

    def __init__(self, kind, who, body=""):
        super().__init__(classes=kind)
        self.kind = kind
        self.who = who
        self.body = body  # the whole text so far
        self.answer_shown = ""  # the part of the answer already handed to the Markdown widget

    def compose(self) -> ComposeResult:
        yield Label(self.who, classes="who")
        if self.kind == "user":
            yield Static(Text(self.body), classes="body")
        else:
            yield Static(Text("…", style="dim"), classes="thought")  # "…" until the first token arrives
            yield Markdown(classes="answer")

    async def add(self, text):
        self.body += text
        thought, answer = split_reply(self.body)

        thought_box = self.query_one(".thought", Static)
        thought_box.update(Text(thought))
        thought_box.display = bool(thought)

        # The answer only ever grows, so hand the Markdown widget just the new part. It re-renders the last
        # block, instead of the whole reply on every token.
        new = answer[len(self.answer_shown) :]
        if new:
            self.answer_shown = answer
            await self.query_one(".answer", Markdown).append(new)


class LyraApp(App):
    TITLE = "lyra"
    CSS = """
    Screen { background: $background; }

    #topbar { height: 1; padding: 0 2; background: $panel; }
    #brand { width: auto; text-style: bold; color: $primary; }
    #topbar-model { width: 1fr; color: $text-muted; padding-left: 2; }
    #status { width: auto; }

    #body { height: 1fr; }
    #sidebar { width: 46; padding: 1 1 0 2; }
    #main { width: 1fr; padding: 1 2 0 1; }

    .panel {
        border: round $panel-lighten-2;
        border-title-color: $text-muted;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
        padding: 0 1;
        background: $background;
    }

    #setup { height: auto; }
    .field-label { color: $text-muted; margin-top: 1; }
    #model { margin-top: 0; }
    #model-info { color: $text-muted; height: 1; margin-bottom: 1; }
    #columns { height: 1; color: $text-muted; }
    #columns .c-role { width: 8; }
    #columns .c-host { width: 1fr; }
    #columns .c-layers { width: 7; }
    #columns .c-x { width: 3; }
    DeviceRow { height: 1; margin-bottom: 1; }
    DeviceRow .role { width: 8; color: $text-muted; }
    DeviceRow .host { width: 1fr; }
    DeviceRow .layers { width: 6; margin-left: 1; }
    DeviceRow .remove { width: 3; min-width: 3; margin-left: 1; color: $text-muted; background: transparent; }
    DeviceRow .remove:hover { color: $error; }
    DeviceRow .spacer { width: 3; margin-left: 1; }
    #device-buttons { height: 1; margin-bottom: 1; }
    #device-buttons Button { width: auto; min-width: 0; padding: 0 1; margin-right: 1; }
    #connect { width: 100%; margin-bottom: 1; }
    #think-row { height: 1; margin-bottom: 1; }
    #think-row Label { width: 1fr; color: $text-muted; }
    #think { width: auto; height: 1; border: none; padding: 0; background: transparent; }

    #log-panel { height: 1fr; margin-top: 1; }
    #log { background: $background; color: $text-muted; scrollbar-size-vertical: 1; }

    #chat-panel { height: 1fr; }
    #chat { scrollbar-size-vertical: 1; padding-top: 1; }
    #empty { color: $text-muted; width: 100%; margin-top: 2; text-align: center; }
    ChatEntry { height: auto; margin-bottom: 1; }
    ChatEntry .who { text-style: bold; }
    ChatEntry.user .who { color: $primary; }
    ChatEntry.assistant .who { color: $success; }
    ChatEntry .body { padding-left: 2; }
    ChatEntry .thought { padding-left: 2; margin-bottom: 1; color: $text-muted; text-style: italic; }
    ChatEntry .answer { padding: 0 0 0 2; margin: 0; background: transparent; }

    #prompt { margin-top: 1; border: round $panel-lighten-2; background: $background; }
    #prompt:focus { border: round $primary; }
    """
    BINDINGS = [
        ("escape", "stop", "Stop reply"),
        ("ctrl+l", "clear_chat", "Clear chat"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.device = os.environ.get("LYRA_DEVICE") or default_device()
        self.me, self.peers = tailnet_hosts()
        self.driver = None
        self.state = "idle"
        self.model_layers = None  # layer count of the model in the Model box, once we know it
        self.messages = []  # the chat history we send to the model
        self.reply_entry = None  # the ChatEntry being streamed into
        self.reply_started = None  # when we sent the prompt
        self.first_token_at = None
        self.last_token_at = None
        self.reply_tokens = 0
        self.pending_text = ""  # tokens that arrived since the last redraw

    def compose(self) -> ComposeResult:
        with HorizontalGroup(id="topbar"):
            yield Label("lyra", id="brand")
            yield Label("", id="topbar-model")
            yield Label("", id="status")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                with VerticalGroup(id="setup", classes="panel") as setup:
                    setup.border_title = "setup"
                    yield Label("model", classes="field-label")
                    yield Input(DEFAULT_MODEL, id="model", placeholder="Hugging Face model id", compact=True)
                    yield Static("", id="model-info")
                    with HorizontalGroup(id="columns"):
                        yield Label("", classes="c-role")
                        yield Label("host", classes="c-host")
                        yield Label("layers", classes="c-layers")
                        yield Label("", classes="c-x")
                    with VerticalGroup(id="devices"):
                        yield DeviceRow(self.me, driver=True, hosts=[self.me])
                        yield DeviceRow(self.peers[0] if self.peers else "", hosts=self.peers)
                    with HorizontalGroup(id="device-buttons"):
                        yield Button("+ add", id="add", compact=True)
                        yield Button("even", id="even", compact=True)
                        yield Button("auto split", id="auto", compact=True)
                    yield Button("connect", id="connect", variant="primary", compact=True)
                    with HorizontalGroup(id="think-row"):
                        yield Label("show thinking")
                        yield Switch(value=True, id="think")
                with Vertical(id="log-panel", classes="panel") as log_panel:
                    log_panel.border_title = "log"
                    yield EventLog(id="log", wrap=True, markup=False, min_width=20)  # wrap to the panel
            with Vertical(id="main"):
                with Vertical(id="chat-panel", classes="panel") as chat_panel:
                    chat_panel.border_title = "chat"
                    with VerticalScroll(id="chat"):
                        yield Static("Set up the devices on the left, then connect.", id="empty")
                yield Input(placeholder="Message…", id="prompt")
        yield Footer()

    def on_mount(self):
        self.theme = "tokyo-night"  # ctrl+p → "theme" to try others
        self.begin_capture_print(self.query_one(EventLog), stdout=True, stderr=True)
        # Redraw the streaming reply 20 times a second instead of once per token. At 100 tok/s, per-token redraws
        # pile up in the event queue, and a keypress like Escape would wait behind all of them.
        self.set_interval(1 / 20, self.flush_reply)
        self.log_event(f"this machine: {self.me} ({self.device})")
        if self.peers:
            self.log_event("online on the tailnet: " + ", ".join(self.peers))
        self.look_up_model(DEFAULT_MODEL)
        self.refresh_plan()

    def log_event(self, text):
        self.query_one(EventLog).write(text)

    def set_stats(self, text):
        self.query_one("#chat-panel").border_subtitle = text

    # ----- the plan -----

    def rows(self):
        return list(self.query(DeviceRow))

    def stages(self):
        """(host, start, count) for each row, in order."""
        stages = []
        start = 0
        for row in self.rows():
            host, count = row.read()
            stages.append((host, start, count))
            start += count
        return stages

    def plan_problem(self):
        """Why the plan can't be used yet, or None if it can."""
        stages = self.stages()
        for i, (host, _, count) in enumerate(stages):
            if not host:
                return "a device has no host"
            if count <= 0 and i > 0:  # the driver may run no layers: it still holds the embedding and head
                return "a worker has no layers"
        assigned = sum(count for _, _, count in stages)
        if self.model_layers is None:
            return "looking up the model…"
        if assigned != self.model_layers:
            return f"{assigned} of {self.model_layers} layers assigned"
        return None

    def refresh_plan(self):
        problem = self.plan_problem()
        info = Text()
        if self.model_layers is not None:
            info.append(f"{self.model_layers} layers · ")
        if problem:
            info.append(problem, style="yellow")
        else:
            info.append("split ok", style="green")
        self.query_one("#model-info", Static).update(info)

        text, colour = STATUS[self.state]
        if self.state == "ready":
            stages = len(self.driver.plan)
            text = f"ready · {stages} stage" if stages == 1 else f"ready · {stages} stages"
        self.query_one("#status", Label).update(f"[{colour}]●[/] {text}")
        self.query_one("#topbar-model", Label).update(self.query_one("#model", Input).value.strip())

        connect = self.query_one("#connect", Button)
        if self.driver is not None:
            connect.label = "disconnect"
            connect.variant = "default"
            connect.disabled = self.state == "busy"
        else:
            connect.label = "connecting…" if self.state == "connecting" else "connect"
            connect.variant = "primary"
            connect.disabled = problem is not None or self.state in ("connecting", "measuring")

    @on(Input.Changed, "DeviceRow Input")
    def device_edited(self):
        self.refresh_plan()

    @on(Button.Pressed, "#add")
    def add_device(self):
        self.query_one("#devices").mount(DeviceRow(hosts=self.peers))
        self.call_after_refresh(self.refresh_plan)

    @on(Button.Pressed, "#even")
    def even_split(self):
        rows = self.rows()
        if not self.model_layers or not rows:
            return
        base, extra = divmod(self.model_layers, len(rows))
        for i, row in enumerate(rows):
            count = base + (1 if i < extra else 0)  # the first devices take the leftover layers
            row.query_one(".layers", Input).value = str(count)
        self.refresh_plan()

    @on(Button.Pressed, "#auto")
    def auto_split_pressed(self):
        addresses = []
        for host, _, _ in self.stages():
            if not host:
                self.notify("Every device needs a host first", severity="warning")
                return
            addresses.append(parse_address(host))
        model = self.query_one("#model", Input).value.strip()
        self.set_state("measuring")
        self.set_devices_editable(False)
        self.run_auto_split(model, addresses)

    @work(thread=True, exclusive=True, group="ring")
    def run_auto_split(self, model, addresses):
        from .autosplit import auto_split

        try:
            self.post_message(AutoSplitDone(auto_split(model, addresses, self.device)))
        except ConnectionRefusedError as e:
            self.post_message(AutoSplitFailed(f"a worker refused the connection ({e}): is `lyra worker` running?"))
        except Exception as e:
            self.post_message(AutoSplitFailed(f"{type(e).__name__}: {e}"))

    async def on_auto_split_done(self, message: AutoSplitDone):
        for i, (row, count) in enumerate(zip(self.rows(), message.counts)):
            if i > 0 and count == 0:
                await row.remove()  # not worth its hop: auto split explained why in the log
            else:
                row.query_one(".layers", Input).value = str(count)
        self.set_devices_editable(True)
        self.set_state("idle")  # also re-checks the plan, now that the unused rows are gone

    def on_auto_split_failed(self, message: AutoSplitFailed):
        self.log_event(f"auto split failed: {message.error}")
        self.set_state("error")
        self.set_devices_editable(True)

    @on(Input.Submitted, "#model")
    def model_submitted(self, event: Input.Submitted):
        self.model_layers = None
        self.refresh_plan()
        self.look_up_model(event.value.strip())

    @work(thread=True, exclusive=True, group="model")
    def look_up_model(self, model):
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model)
            self.post_message(ModelInfo(model, config.get_text_config().num_hidden_layers))
        except Exception as e:  # a typo in the model id, no network...
            self.post_message(ModelInfo(model, None, error=str(e).splitlines()[0]))

    def on_model_info(self, message: ModelInfo):
        if message.model != self.query_one("#model", Input).value.strip():
            return  # the box changed since we started looking
        if message.error:
            self.log_event(f"can't load {message.model}: {message.error}")
        self.model_layers = message.layers
        self.refresh_plan()

    # ----- connecting -----

    @on(Button.Pressed, "#connect")
    def connect_pressed(self):
        if self.driver is not None:
            self.disconnect()
            return
        plan_args = []
        for host, _, count in self.stages():
            plan_args.append(f"{host}={count}")
        model = self.query_one("#model", Input).value.strip()
        self.set_state("connecting")
        self.log_event("connecting: " + " ".join(plan_args))
        self.set_devices_editable(False)
        self.start_ring(model, parse_plan(plan_args))

    @work(thread=True, exclusive=True, group="ring")
    def start_ring(self, model, plan):
        try:
            self.post_message(RingReady(Driver(model, plan, self.device)))
        except ConnectionRefusedError:
            first_worker = plan[1]
            self.post_message(
                RingFailed(
                    f"{first_worker.host}:{first_worker.port} refused the connection: is `lyra worker` running there?"
                )
            )
        except Exception as e:
            self.post_message(RingFailed(f"{type(e).__name__}: {e}"))

    def on_ring_ready(self, message: RingReady):
        self.driver = message.driver
        self.set_state("ready")
        for empty in self.query("#empty"):  # gone once the chat has messages in it
            empty.update("Connected. Type a message below.")
        self.query_one("#prompt", Input).focus()

    async def on_ring_failed(self, message: RingFailed):
        self.log_event(message.error)
        if self.driver is not None:  # the ring broke mid-chat: this driver is no use any more
            await self.flush_reply()
            self.driver.close()
            self.driver = None
        self.set_state("error")
        self.set_devices_editable(True)

    def disconnect(self):
        self.driver.close()  # the workers see the ring close and go back to waiting
        self.driver = None
        gc.collect()
        torch.accelerator.empty_cache()
        self.log_event("disconnected")
        self.set_state("idle")
        self.set_devices_editable(True)

    def set_devices_editable(self, editable):
        for widget in self.query("DeviceRow Input, DeviceRow Button, #add, #even, #auto, #model"):
            widget.disabled = not editable

    def set_state(self, state):
        self.state = state
        self.refresh_plan()

    # ----- chatting -----

    @on(Input.Submitted, "#prompt")
    def prompt_submitted(self, event: Input.Submitted):
        prompt = event.value.strip()
        if not prompt:
            return
        if self.driver is None:
            self.notify("Connect to the ring first", severity="warning")
            return
        if self.state == "busy":
            return
        event.input.value = ""
        self.messages.append({"role": "user", "content": prompt})
        for empty in self.query("#empty"):
            empty.remove()
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(ChatEntry("user", "you", prompt))
        model_name = self.driver.model_name.split("/")[-1]  # "Qwen/Qwen3-8B" -> "Qwen3-8B"
        self.reply_entry = ChatEntry("assistant", model_name)
        chat.mount(self.reply_entry)
        chat.scroll_end(animate=False)

        self.reply_started = time.perf_counter()
        self.first_token_at = None
        self.reply_tokens = 0
        self.set_state("busy")
        self.set_stats("prefill…")
        self.generate(self.query_one("#think", Switch).value)

    @work(thread=True, exclusive=True, group="generate")
    def generate(self, think):
        def on_token(text):
            self.post_message(TokenArrived(text, time.perf_counter()))

        try:
            reply = self.driver.generate(self.messages, on_token=on_token, think=think)
        except Exception as e:  # usually a worker went away
            self.post_message(RingFailed(f"generation failed: {type(e).__name__}: {e}"))
            return
        self.post_message(ReplyDone(reply, self.driver.last_stats))

    def on_token_arrived(self, message: TokenArrived):
        self.reply_tokens += 1
        if self.first_token_at is None:
            self.first_token_at = message.when  # the first token comes out of the prefill
        self.last_token_at = message.when
        self.pending_text += message.text  # flush_reply draws it

    async def flush_reply(self):
        if not self.pending_text:
            return
        text = self.pending_text
        self.pending_text = ""  # before the await: tokens that arrive meanwhile go into the next flush
        await self.reply_entry.add(text)
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

        prefill_ms = (self.first_token_at - self.reply_started) * 1000
        stats = f"prefill {prefill_ms:.0f} ms"
        decode_tokens = self.reply_tokens - 1
        elapsed = self.last_token_at - self.first_token_at
        if decode_tokens > 0 and elapsed > 0:
            stats += f" · {decode_tokens / elapsed:.1f} tok/s"
        self.set_stats(stats)

    async def on_reply_done(self, message: ReplyDone):
        await self.flush_reply()  # draw any tokens that came in since the last redraw
        self.messages.append({"role": "assistant", "content": message.reply})
        s = message.stats
        self.set_stats(
            f"prefill {s['prompt_tokens']} tokens in {s['prefill_ms']:.0f} ms · "
            f"{s['decode_tokens']} tokens at {s['tok_per_s']:.1f} tok/s"
        )
        self.set_state("ready")

    def action_stop(self):
        if self.driver is not None and self.state == "busy":
            self.driver.stop_requested = True

    def action_clear_chat(self):
        if self.state == "busy":
            return
        self.messages = []
        chat = self.query_one("#chat", VerticalScroll)
        chat.remove_children()
        chat.mount(Static("Chat cleared.", id="empty"))
        self.set_stats("")


def main():
    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils import logging

    disable_progress_bars()  # progress bars redraw with \r, which turns into noise in the log panel
    logging.set_verbosity_error()
    LyraApp().run()
    # A background thread may still be blocked on a socket (e.g. waiting for a worker that never answered).
    # Exiting the process closes everything, and the workers see the ring close.
    os._exit(0)


if __name__ == "__main__":
    main()
