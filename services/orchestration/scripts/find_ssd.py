#!/usr/bin/env python3
"""
Find and display information about mounted SSDs/external drives on Linux.

Usage:
    python scripts/find_ssd.py          # Show SSD discovery report
    python scripts/find_ssd.py --open   # Show command to access the SSD
"""
import argparse
import subprocess
import sys
from pathlib import Path


# Mountpoints to exclude (system partitions)
EXCLUDE_MOUNTS = {"/", "/boot", "/boot/efi", "/home", "/var", "/tmp", "/snap"}
EXCLUDE_PREFIXES = ("/snap/", "/boot/")

# Preferred mount locations for external/secondary drives
PREFERRED_PREFIXES = ("/mnt/", "/media/", "/run/media/")


def run_cmd(cmd: list[str]) -> str:
    """Run a command and return stdout."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running {' '.join(cmd)}: {e}"


def get_mount_info() -> list[dict]:
    """Parse df -h output to get mount information."""
    output = run_cmd(["df", "-h"])
    mounts = []
    
    for line in output.split("\n")[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 6:
            device = parts[0]
            size = parts[1]
            used = parts[2]
            avail = parts[3]
            use_pct = parts[4]
            mountpoint = " ".join(parts[5:])  # Handle spaces in mountpoint
            
            mounts.append({
                "device": device,
                "size": size,
                "used": used,
                "avail": avail,
                "use_pct": use_pct,
                "mountpoint": mountpoint,
            })
    
    return mounts


def get_filesystem_info() -> dict[str, str]:
    """Parse lsblk -f to get filesystem types."""
    output = run_cmd(["lsblk", "-f", "-o", "NAME,FSTYPE,MOUNTPOINT", "-n"])
    fs_map = {}
    
    for line in output.split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0].lstrip("├─└─│ ")
            fstype = parts[1] if len(parts) >= 2 else ""
            mountpoint = parts[2] if len(parts) >= 3 else ""
            if mountpoint:
                fs_map[mountpoint] = fstype
    
    return fs_map


def find_ssd_candidates(mounts: list[dict], fs_info: dict[str, str]) -> list[dict]:
    """Filter mounts to find likely SSD/external drive candidates."""
    candidates = []
    
    for m in mounts:
        mp = m["mountpoint"]
        
        # Skip system mounts
        if mp in EXCLUDE_MOUNTS:
            continue
        if any(mp.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        
        # Skip pseudo-filesystems
        device = m["device"]
        if device.startswith("tmpfs") or device.startswith("overlay"):
            continue
        
        # Add filesystem type
        m["fstype"] = fs_info.get(mp, "unknown")
        
        # Score: prefer /mnt, /media, /run/media mounts
        m["preferred"] = any(mp.startswith(prefix) for prefix in PREFERRED_PREFIXES)
        
        candidates.append(m)
    
    # Sort: preferred mounts first, then by size (descending)
    def sort_key(x):
        # Parse size (e.g., "1.8T", "500G", "100M")
        size_str = x["avail"]
        try:
            num = float(size_str[:-1])
            unit = size_str[-1].upper()
            multiplier = {"K": 1, "M": 1e3, "G": 1e6, "T": 1e9}.get(unit, 1)
            size_val = num * multiplier
        except:
            size_val = 0
        return (not x["preferred"], -size_val)
    
    candidates.sort(key=sort_key)
    return candidates


def list_mount_dirs() -> list[str]:
    """List directories under /mnt, /media, /run/media."""
    dirs = []
    for base in ["/mnt", "/media", "/run/media"]:
        base_path = Path(base)
        if base_path.exists():
            try:
                for item in base_path.iterdir():
                    if item.is_dir():
                        dirs.append(str(item))
                        # Also check subdirs (e.g., /run/media/username/drive)
                        try:
                            for subitem in item.iterdir():
                                if subitem.is_dir():
                                    dirs.append(str(subitem))
                        except PermissionError:
                            pass
            except PermissionError:
                pass
    return sorted(set(dirs))


def main():
    parser = argparse.ArgumentParser(description="Find mounted SSDs/external drives")
    parser.add_argument("--open", action="store_true", help="Show command to access the SSD")
    args = parser.parse_args()
    
    print()
    print("=" * 70)
    print("SSD / EXTERNAL DRIVE DISCOVERY")
    print("=" * 70)
    
    # Section 1: lsblk -f
    print()
    print("─" * 70)
    print("BLOCK DEVICES (lsblk -f)")
    print("─" * 70)
    print(run_cmd(["lsblk", "-f"]))
    
    # Section 2: df -h
    print()
    print("─" * 70)
    print("DISK USAGE (df -h)")
    print("─" * 70)
    print(run_cmd(["df", "-h"]))
    
    # Section 3: Mount directories
    print()
    print("─" * 70)
    print("MOUNT DIRECTORIES (/mnt, /media, /run/media)")
    print("─" * 70)
    mount_dirs = list_mount_dirs()
    if mount_dirs:
        for d in mount_dirs:
            print(f"  {d}")
    else:
        print("  (no directories found)")
    
    # Section 4: SSD candidates
    print()
    print("─" * 70)
    print("LIKELY SSD / EXTERNAL DRIVE CANDIDATES")
    print("─" * 70)
    
    mounts = get_mount_info()
    fs_info = get_filesystem_info()
    candidates = find_ssd_candidates(mounts, fs_info)
    
    if candidates:
        print(f"  {'MOUNTPOINT':<40} {'FSTYPE':<10} {'SIZE':<8} {'AVAIL':<8} {'USE%':<6}")
        print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
        for c in candidates:
            star = "★" if c["preferred"] else " "
            print(f"{star} {c['mountpoint']:<40} {c['fstype']:<10} {c['size']:<8} {c['avail']:<8} {c['use_pct']:<6}")
    else:
        print("  (no candidates found)")
    
    # Section 5: Best guess
    print()
    print("─" * 70)
    print("BEST GUESS")
    print("─" * 70)
    
    if candidates:
        best = candidates[0]
        ssd_root = best["mountpoint"]
        print(f"  SSD_ROOT={ssd_root}")
        print()
        print(f"  Device:     {best['device']}")
        print(f"  Filesystem: {best['fstype']}")
        print(f"  Size:       {best['size']} (Available: {best['avail']})")
        
        if args.open:
            print()
            print("─" * 70)
            print("ACCESS COMMAND")
            print("─" * 70)
            print(f'  cd "{ssd_root}" && ls -la')
            print()
            print("  Copy and paste the above command to access your SSD.")
    else:
        print("  No external/secondary drives detected.")
        print()
        print("  Tips:")
        print("    - Check if the drive is connected")
        print("    - Try: sudo mount /dev/sdX1 /mnt/ssd")
        print("    - Check dmesg for device detection: dmesg | tail -20")
    
    print()
    print("=" * 70)
    
    # Export-friendly output
    if candidates:
        print()
        print("# Export for shell:")
        print(f"export SSD_ROOT=\"{candidates[0]['mountpoint']}\"")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
