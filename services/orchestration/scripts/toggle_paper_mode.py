#!/usr/bin/env python3
"""
Toggle paper trading mode in config/app.yaml.

Usage:
    python scripts/toggle_paper_mode.py --on   # Enable paper trading
    python scripts/toggle_paper_mode.py --off  # Disable paper trading
    python scripts/toggle_paper_mode.py        # Show current status
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CONFIG_FILE = Path(__file__).parent.parent / "config" / "app.yaml"


def get_current_mode() -> bool:
    """Get current execution_dry_run value from config."""
    if not CONFIG_FILE.exists():
        print(f"❌ Config file not found: {CONFIG_FILE}")
        sys.exit(1)
    
    content = CONFIG_FILE.read_text()
    
    # Look for execution_dry_run: true/false (not in comments)
    # Match lines that don't start with # or have # before the key
    for line in content.split("\n"):
        # Skip comment lines
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        
        match = re.search(r"execution_dry_run:\s*(true|false)", line, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
    
    # Default is true if not found
    return True


def set_mode(dry_run: bool) -> None:
    """Set execution_dry_run value in config."""
    if not CONFIG_FILE.exists():
        print(f"❌ Config file not found: {CONFIG_FILE}")
        sys.exit(1)
    
    content = CONFIG_FILE.read_text()
    new_value = "true" if dry_run else "false"
    
    # Find and replace only the actual setting, not comments
    lines = content.split("\n")
    new_lines = []
    replaced = False
    
    for line in lines:
        stripped = line.lstrip()
        # Skip comment lines - keep as-is
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        
        # Check for the setting
        if "execution_dry_run:" in line and not replaced:
            # Replace the value, preserving the rest of the line (comments)
            new_line = re.sub(
                r"(execution_dry_run:\s*)(true|false)",
                rf"\g<1>{new_value}",
                line,
                flags=re.IGNORECASE,
            )
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)
    
    if not replaced:
        print(f"❌ Could not find 'execution_dry_run' in {CONFIG_FILE}")
        sys.exit(1)
    
    CONFIG_FILE.write_text("\n".join(new_lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Toggle paper trading mode")
    parser.add_argument("--on", action="store_true", help="Enable paper trading (dry_run=false)")
    parser.add_argument("--off", action="store_true", help="Disable paper trading (dry_run=true)")
    args = parser.parse_args()
    
    current = get_current_mode()
    
    if args.on and args.off:
        print("❌ Cannot specify both --on and --off")
        return 1
    
    if args.on:
        # Enable paper trading = set dry_run to FALSE
        if not current:
            print("✅ Paper trading is already ENABLED (execution_dry_run: false)")
        else:
            set_mode(dry_run=False)
            print("✅ Paper trading ENABLED")
            print("   execution_dry_run: false")
            print("\n💡 Don't forget to also run: make approve")
        return 0
    
    if args.off:
        # Disable paper trading = set dry_run to TRUE
        if current:
            print("✅ Paper trading is already DISABLED (execution_dry_run: true)")
        else:
            set_mode(dry_run=True)
            print("🛑 Paper trading DISABLED")
            print("   execution_dry_run: true")
        return 0
    
    # Show current status
    print("\n" + "=" * 50)
    print("📊 Paper Trading Status")
    print("=" * 50)
    
    if current:
        print("\n🛑 Paper trading is DISABLED")
        print("   execution_dry_run: true")
        print("   All trades will be BLOCKED (no Alpaca API calls)")
        print("\n💡 To enable: make paper-on")
    else:
        print("\n✅ Paper trading is ENABLED")
        print("   execution_dry_run: false")
        print("   Trades will be submitted to Alpaca Paper API")
        print("\n💡 To disable: make paper-off")
    
    # Check approval file
    approval_file = Path(__file__).parent.parent / ".tmp" / "APPROVE_EXECUTION"
    print("\n" + "-" * 50)
    if approval_file.exists():
        print(f"✅ Approval file EXISTS: {approval_file}")
    else:
        print(f"⚠️  Approval file MISSING: {approval_file}")
        if not current:
            print("   ⚠️  Trades will still be BLOCKED until you run: make approve")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
