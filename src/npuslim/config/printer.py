"""Config pretty printer."""
from rich.console import Console
from rich.table import Table

from npuslim.config.schema import EngineConfig


def print_config(config: EngineConfig, console: Console = None) -> None:
    """
    Pretty print EngineConfig using rich tables.

    Args:
        config: Configuration to print
        console: Rich console (defaults to stdout)
    """
    console = console or Console()

    # Metadata
    if config.metadata.name or config.metadata.description:
        console.print(f"[bold blue]Metadata[/bold blue]")
        if config.metadata.name:
            console.print(f"  Name: {config.metadata.name}")
        if config.metadata.description:
            console.print(f"  Description: {config.metadata.description}")
        console.print()

    # Resources table
    if config.resources:
        table = Table(title="[bold green]Resources[/bold green]")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Extra", style="dim")

        for r in config.resources:
            extra = ", ".join(f"{k}={v}" for k, v in r.extra.items())
            table.add_row(r.id, r.type, extra or "-")

        console.print(table)
        console.print()

    # Recipe table
    if config.recipe:
        table = Table(title="[bold magenta]Recipe[/bold magenta]")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Model", style="green")
        table.add_column("Data", style="green")
        table.add_column("Algorithm", style="blue")

        for t in config.recipe:
            model_ref = t.model or t.main_model or "-"
            algo = t.algorithm.type if t.algorithm else "-"
            table.add_row(t.name, t.type, model_ref, t.data or "-", algo)

        console.print(table)

        # Algorithm details
        for t in config.recipe:
            if t.algorithm and t.algorithm.extra:
                console.print(f"\n[bold]{t.name}[/bold] algorithm: {t.algorithm.extra}")
