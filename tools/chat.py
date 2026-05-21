#!/usr/bin/env python3
"""Interactive chat client for vLLM OpenAI-compatible API."""

import sys
import argparse
from openai import OpenAI

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.theme import Theme

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

CUSTOM_THEME = Theme(
    {
        "user": "bold cyan",
        "assistant": "bold green",
        "system": "bold yellow",
        "info": "dim",
        "error": "bold red",
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with vLLM API")
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", default="/mnt/xufan_400T/models/LongCat-Flash-Chat")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--chat", action="store_true",
                        help="Use chat completions API (applies model's chat template)")
    parser.add_argument("--no-history", action="store_true", help="Disable multi-turn history")
    return parser.parse_args()


def detect_model(client):
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return None


def stream_completion(client, model, prompt, args, console):
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=True,
        )
        full_text = ""
        if HAS_RICH:
            console.print("[assistant]Assistant:[/assistant]")
            for chunk in response:
                text = chunk.choices[0].text or ""
                full_text += text
                console.print(text, end="", highlight=False)
            console.print()
        else:
            print("Assistant: ", end="", flush=True)
            for chunk in response:
                text = chunk.choices[0].text or ""
                full_text += text
                print(text, end="", flush=True)
            print()
        return full_text
    except Exception as e:
        msg = f"Request failed: {e}"
        if HAS_RICH:
            console.print(msg, style="error")
        else:
            print(f"ERROR: {msg}")
        return None


def stream_chat(client, model, messages, args, console):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=True,
        )
        full_text = ""
        if HAS_RICH:
            console.print("[assistant]Assistant:[/assistant]")
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                full_text += delta
                console.print(delta, end="", highlight=False)
            console.print()
        else:
            print("Assistant: ", end="", flush=True)
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                full_text += delta
                print(delta, end="", flush=True)
            print()
        return full_text
    except Exception as e:
        msg = f"Request failed: {e}"
        if HAS_RICH:
            console.print(msg, style="error")
        else:
            print(f"ERROR: {msg}")
        return None


def print_help(console):
    help_text = (
        "[bold]Commands:[/bold]\n"
        "  /clear   Clear screen and history\n"
        "  /exit    Exit chat\n"
        "  /help    Show this help\n"
        "  /model   Show current model\n"
        "  /temp T  Set temperature\n"
        "  /topp P  Set top-p"
    )
    if HAS_RICH:
        console.print(Panel(help_text, title="Help", style="system"))
    else:
        print(help_text)


def main():
    args = parse_args()

    if HAS_RICH:
        console = Console(theme=CUSTOM_THEME)
    else:
        console = None

    client = OpenAI(base_url=args.url, api_key="empty")

    model = args.model or detect_model(client)
    if not model:
        msg = "No model found. Specify --model."
        if HAS_RICH:
            console.print(msg, style="error")
        else:
            print(f"ERROR: {msg}")
        sys.exit(1)

    history = []

    banner = (
        f"[bold]vLLM Chat[/bold]\n"
        f"  Model:   {model}\n"
        f"  API:     {args.url}\n"
        f"  Mode:    {'chat' if args.chat else 'completion'}\n"
        f"  Temp:    {args.temperature}  |  Top-P: {args.top_p}  |  Max tokens: {args.max_tokens}\n"
        f"  Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit"
    )
    if HAS_RICH:
        console.print(Panel(banner, style="system", title="vLLM Chat", title_align="left"))
    else:
        print(banner)

    while True:
        try:
            if HAS_RICH:
                user_input = Prompt.ask("\n[user]You")
            else:
                user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/exit", "/quit", "/q"):
                break
            elif cmd[0] == "/clear":
                history = []
                if HAS_RICH:
                    console.clear()
                else:
                    print("\033[2J\033[H", end="")
                continue
            elif cmd[0] == "/help":
                print_help(console)
                continue
            elif cmd[0] == "/model":
                msg = f"Current model: {model}"
                if HAS_RICH:
                    console.print(msg, style="info")
                else:
                    print(msg)
                continue
            elif cmd[0] == "/temp" and len(cmd) > 1:
                try:
                    args.temperature = float(cmd[1])
                    msg = f"Temperature set to {args.temperature}"
                except ValueError:
                    msg = "Invalid value"
                if HAS_RICH:
                    console.print(msg, style="info")
                else:
                    print(msg)
                continue
            elif cmd[0] == "/topp" and len(cmd) > 1:
                try:
                    args.top_p = float(cmd[1])
                    msg = f"Top-P set to {args.top_p}"
                except ValueError:
                    msg = "Invalid value"
                if HAS_RICH:
                    console.print(msg, style="info")
                else:
                    print(msg)
                continue
            else:
                msg = f"Unknown command: {user_input}"
                if HAS_RICH:
                    console.print(msg, style="error")
                else:
                    print(msg)
                continue

        if args.chat:
            history.append({"role": "user", "content": user_input})
            messages = history if not args.no_history else [{"role": "user", "content": user_input}]
            reply = stream_chat(client, model, messages, args, console)
            if reply:
                history.append({"role": "assistant", "content": reply})
        else:
            reply = stream_completion(client, model, user_input, args, console)

    farewell = "Goodbye!"
    if HAS_RICH:
        console.print(f"\n[system]{farewell}")
    else:
        print(f"\n{farewell}")


if __name__ == "__main__":
    main()
