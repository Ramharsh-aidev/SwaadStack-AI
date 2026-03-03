"""Shared helper functions — timing, serialization, console display."""

import json
import time
import functools
from typing import Any, Dict

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from swaadstack.utils.logging import logger

console = Console()


# ==============================================================================
# Performance Timing
# ==============================================================================
def timeit(func):
    """Decorator to measure and log function execution time in ms."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("function_executed", function=func.__name__, elapsed_ms=round(elapsed_ms, 2))
        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("async_function_executed", function=func.__name__, elapsed_ms=round(elapsed_ms, 2))
        return result

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


# ==============================================================================
# Serialization
# ==============================================================================
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def save_json(data: Any, filepath: str):
    """Save data to JSON file with numpy support."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# Console Display
# ==============================================================================
def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def print_banner(title: str):
    """Print a styled section banner."""
    console.print(f"\n{'=' * 60}", style="bold cyan")
    console.print(f"  🍛 {title}", style="bold yellow")
    console.print(f"{'=' * 60}\n", style="bold cyan")


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty-print evaluation metrics."""
    console.print(f"\n📊 [bold]{title}[/bold]")
    for key, value in metrics.items():
        color = "green" if value > 0.7 else "yellow" if value > 0.4 else "red"
        console.print(f"   {key}: [{color}]{value:.4f}[/{color}]")
