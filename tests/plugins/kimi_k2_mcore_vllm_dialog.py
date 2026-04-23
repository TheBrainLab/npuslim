#!/usr/bin/env python3
"""Terminal chat client for Kimi K2 MCore on vLLM, aligned with forward path.

Design choices:
- /v1/completions (not chat/completions)
- Prompt format per turn: "[BOS]" + user_text
- Stateless: each question is independent (no history memory)
- temperature=0, top_p=1, add_special_tokens=False
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterator
import urllib.error
import urllib.request

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


console = Console()


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_sse_data(url: str, payload: dict, timeout: float) -> Iterator[str]:
    """Yield `data:` payloads from an SSE response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        event_lines: list[str] = []
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if event_lines:
                    data_parts = []
                    for item in event_lines:
                        if item.startswith("data:"):
                            data_parts.append(item[5:].lstrip())
                    if data_parts:
                        yield "\n".join(data_parts)
                    event_lines = []
                continue
            if line.startswith(":"):
                continue
            event_lines.append(line)
        if event_lines:
            data_parts = []
            for item in event_lines:
                if item.startswith("data:"):
                    data_parts.append(item[5:].lstrip())
            if data_parts:
                yield "\n".join(data_parts)


def _http_get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_model(base_url: str, timeout: float) -> str:
    models = _http_get_json(f"{base_url}/v1/models", timeout=timeout)
    data = models.get("data", [])
    if not data:
        raise RuntimeError("No model returned from /v1/models")
    return data[0]["id"]


def build_prompt(user_text: str) -> str:
    # Keep BOS explicit to align with forward-debug style.
    return "[BOS]" + user_text


def _build_payload(model: str, prompt: str, max_tokens: int, stream: bool) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "return_token_ids": True,
        "stream": stream,
    }


def generate_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[str, list[int] | None]:
    payload = _build_payload(model=model, prompt=prompt, max_tokens=max_tokens, stream=False)
    out = _http_post_json(f"{base_url}/v1/completions", payload=payload, timeout=timeout)
    choices = out.get("choices", [])
    if not choices:
        raise RuntimeError(f"No choices in response: {out}")
    choice0 = choices[0]
    text = choice0.get("text", "")
    token_ids = choice0.get("token_ids")
    return text, token_ids


def generate_once_stream(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[str, list[int] | None]:
    payload = _build_payload(model=model, prompt=prompt, max_tokens=max_tokens, stream=True)
    text_parts: list[str] = []
    last_token_ids: list[int] | None = None
    saw_text = False

    for data in _http_post_sse_data(
        f"{base_url}/v1/completions",
        payload=payload,
        timeout=timeout,
    ):
        if data == "[DONE]":
            break
        obj = json.loads(data)
        choices = obj.get("choices", [])
        if not choices:
            continue
        choice0 = choices[0]
        delta = choice0.get("text", "")
        if delta:
            saw_text = True
            text_parts.append(delta)
            console.print(delta, end="", highlight=False, markup=False, soft_wrap=True)
        token_ids = choice0.get("token_ids")
        if isinstance(token_ids, list):
            last_token_ids = token_ids

    if not saw_text:
        console.print("(empty)", style="dim")
    else:
        console.print()
    return "".join(text_parts), last_token_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kimi K2 MCore dialog over vLLM completions (forward-consistent, stateless)."
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default=None, help="Optional; defaults to first /v1/models id")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming and wait for full completion.",
    )
    p.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Print generated token IDs for debugging.",
    )
    return p.parse_args()


def render_header(*, base_url: str, model: str, streaming_on: bool) -> None:
    console.clear()
    console.print(
        Panel.fit(
            "[bold cyan]Kimi K2 MCore Terminal Chat[/bold cyan]\n"
            "[dim]Forward-consistent mode (stateless per turn)[/dim]",
            border_style="cyan",
        )
    )
    console.print(f"[dim]API:[/dim] {base_url}/v1/completions")
    console.print(f"[dim]Model:[/dim] {model}")
    console.print("[dim]Prompt rule:[/dim] [BOS] + user_text")
    console.print("[dim]Sampling:[/dim] temperature=0, top_p=1")
    console.print(f"[dim]Streaming:[/dim] {'on' if streaming_on else 'off'}")
    console.print("[dim]Commands:[/dim] /clear, /exit, /quit")


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        model = args.model or resolve_model(base_url, timeout=args.timeout)
    except Exception as exc:
        console.print(f"[red]Failed to resolve model:[/red] {exc}", file=sys.stderr)
        return 1

    render_header(base_url=base_url, model=model, streaming_on=not args.no_stream)

    while True:
        try:
            user_text = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/exit", "/quit"}:
            break
        if command == "/clear":
            render_header(base_url=base_url, model=model, streaming_on=not args.no_stream)
            continue

        prompt = build_prompt(user_text)
        try:
            if args.no_stream:
                text, token_ids = generate_once(
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            else:
                console.print("[bold green]Assistant[/bold green]")
                text, token_ids = generate_once_stream(
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            console.print(f"[red]HTTP {exc.code} {exc.reason}[/red]")
            console.print(body, style="dim")
            continue
        except Exception as exc:
            console.print(f"[red]Request failed:[/red] {exc}")
            continue

        if args.no_stream:
            assistant = text.strip()
            console.print("[bold green]Assistant[/bold green]")
            console.print(Panel(assistant if assistant else "(empty)", border_style="green"))
        if args.show_token_ids:
            console.print(f"[dim]token_ids:[/dim] {token_ids}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
