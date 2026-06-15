"""Colored console output for alerts and analysis summaries."""

import sys
from datetime import datetime
from typing import Optional

from src.models.analysis import AnalysisResultBundle
from src.models.alert import Alert


# ANSI color codes
COLORS = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def print_alert(alert: Alert) -> None:
    """Print a formatted alert to console with color coding.

    Args:
        alert: Alert to display.
    """
    level_colors = {
        "critical": "red",
        "warning": "yellow",
        "info": "blue",
    }
    level_icons = {
        "critical": "!!!",
        "warning": "!!",
        "info": "i",
    }

    color = level_colors.get(alert.level, "reset")
    icon = level_icons.get(alert.level, "?")

    print()
    print(f"{COLORS[color]}{COLORS['bold']}"
          f"[{icon}] {alert.title}"
          f"{COLORS['reset']}")
    print(f"{COLORS[color]}  Level: {alert.level.upper()} | Rule: {alert.rule_name}"
          f"{COLORS['reset']}")
    print(f"  {alert.detail}")
    if alert.remediation_plan:
        print(f"\n  {COLORS['green']}Remediation Plan:{COLORS['reset']}")
        for line in alert.remediation_plan.split("\n"):
            print(f"  {line}")
    print(f"  {COLORS['blue']}Alert ID: {alert.id[:8]}{COLORS['reset']}")
    print()


def print_summary(bundle: AnalysisResultBundle) -> None:
    """Print a compact analysis summary to console.

    Args:
        bundle: Complete analysis result bundle.
    """
    total = bundle.total_reviews
    valid = bundle.valid_reviews

    print()
    print(f"{COLORS['bold']}{'='*60}{COLORS['reset']}")
    print(f"{COLORS['bold']}  E-commerce Review Analysis Report{COLORS['reset']}")
    print(f"{COLORS['bold']}{'='*60}{COLORS['reset']}")
    print(f"  Product ID: {bundle.product_id or 'All Products'}")
    print(f"  Analysis Time: {bundle.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'─'*56}")

    # Metrics
    print(f"  Total Reviews:   {total:>6}")
    print(f"  Valid Reviews:   {valid:>6} ({_pct(valid, total)})")
    print(f"  {COLORS['green']}Positive:        {bundle.positive_count:>6} ({_pct(bundle.positive_count, valid)}){COLORS['reset']}")
    print(f"  {COLORS['blue']}Neutral:         {bundle.neutral_count:>6} ({_pct(bundle.neutral_count, valid)}){COLORS['reset']}")
    print(f"  {COLORS['red']}Negative:        {bundle.negative_count:>6} ({_pct(bundle.negative_count, valid)}){COLORS['reset']}")
    print(f"  {'─'*56}")

    # Top pain points
    if bundle.pain_points:
        print(f"  {COLORS['bold']}Top Pain Points:{COLORS['reset']}")
        for pp in bundle.pain_points[:5]:
            icon = "🔴" if pp.severity == "high" else "🟡" if pp.severity == "medium" else "🟢"
            hf = " [HIGH FREQ]" if pp.is_high_frequency else ""
            print(f"    {icon} [{pp.category}] {pp.description} ({pp.frequency} mentions){hf}")

    # Keywords
    if bundle.keywords:
        print(f"  {'─'*56}")
        top_keywords = [k.keyword for k in bundle.keywords[:10]]
        print(f"  {COLORS['bold']}Top Keywords:{COLORS['reset']} {', '.join(top_keywords)}")

    # Alerts
    if bundle.alerts:
        print(f"  {'─'*56}")
        print(f"  {COLORS['red']}{COLORS['bold']}⚠ Alerts: {len(bundle.alerts)}{COLORS['reset']}")
        for alert in bundle.alerts:
            icon = "🔴" if alert.level == "critical" else "🟡" if alert.level == "warning" else "🔵"
            print(f"    {icon} [{alert.level.upper()}] {alert.title}")

    # Anomalies
    if bundle.anomaly_events:
        print(f"  {'─'*56}")
        print(f"  {COLORS['yellow']}Anomalies Detected: {len(bundle.anomaly_events)}{COLORS['reset']}")
        for event in bundle.anomaly_events[:5]:
            print(f"    • {event.description[:120]}")

    print(f"{COLORS['bold']}{'='*60}{COLORS['reset']}")
    print()


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{COLORS['red']}[ERROR] {message}{COLORS['reset']}", file=sys.stderr)


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{COLORS['yellow']}[WARN] {message}{COLORS['reset']}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{COLORS['blue']}[INFO] {message}{COLORS['reset']}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{COLORS['green']}[OK] {message}{COLORS['reset']}")


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{part / total:.1%}"
