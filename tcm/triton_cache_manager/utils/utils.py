"""
Utilities.
"""

import re
from typing import Optional
from datetime import timedelta, datetime
import rich
import typer


def format_size(size_bytes: int) -> str:
    """
    Format a file size in a human-readable way.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def parse_duration(duration_str: Optional[str]) -> Optional[timedelta]:
    """
    Parses a duration string (e.g., '7d', '2w') into a timedelta object.
    Returns None if the string is invalid or None.
    """
    if not duration_str:
        return None

    match = re.match(r"(\d+)([dw])$", duration_str.lower())
    if not match:
        rich.print(
            f"[red]Invalid duration format: '{duration_str}'. "
            f"Use 'Xd' for days or 'Xw' for weeks.[/red]"
        )
        raise typer.Exit(code=1)

    value, unit = match.groups()
    value = int(value)

    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    return None


def mod_time_handle(mod_time_unix) -> str:
    """
    Convert an optional UNIX timestamp into a formatted date string.

    Args:
        timestamp: An optional float representing the UNIX timestamp.

    Returns:
        A string formatted as 'YYYY-MM-DD HH:MM:SS',
        'Invalid Date' if the timestamp causes an error during conversion,
        or 'N/A' if the timestamp is None.
    """
    if mod_time_unix is not None:
        try:
            dt_obj = datetime.fromtimestamp(mod_time_unix)
            return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError, OSError):
            return "Invalid Date"
    return "N/A"
