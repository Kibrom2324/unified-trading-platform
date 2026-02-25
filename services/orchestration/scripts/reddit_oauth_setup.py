#!/usr/bin/env python3
"""
Reddit OAuth Setup Helper - Validates Reddit API credentials.

This script helps you configure and test Reddit API credentials
for a "script" type app (personal use).

=== Sample .env.example ===
# Reddit API Credentials (get from https://www.reddit.com/prefs/apps)
# Create a "script" type app, then copy these values:
REDDIT_CLIENT_ID=your_14_char_client_id      # Below app name, looks like: Ab1Cd2Ef3Gh4Ij
REDDIT_CLIENT_SECRET=your_27_char_secret     # The "secret" field
REDDIT_USERNAME=your_reddit_username         # Your Reddit username (without u/)
REDDIT_PASSWORD=your_reddit_password         # Your Reddit password
REDDIT_USER_AGENT=AWET/1.0 (by /u/your_username)  # Custom user agent

=== How to create a Reddit App ===
1. Go to: https://www.reddit.com/prefs/apps
2. Scroll to bottom, click "create app" or "create another app"
3. Fill in:
   - Name: AWET Trading Bot (do NOT include "reddit" in name)
   - Type: script (for personal use scripts)
   - Description: (optional)
   - About URL: (leave blank)
   - Redirect URI: http://localhost:8080 (required but not used for script apps)
4. Click "create app"
5. Copy:
   - client_id: The string under the app name (14 characters)
   - secret: The "secret" field value

=== Usage ===
    python scripts/reddit_oauth_setup.py

"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

# Load .env if present
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"

def load_dotenv():
    """Load environment variables from .env file."""
    if not ENV_FILE.exists():
        return
    
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ.setdefault(key, value)


def get_credentials() -> dict[str, str]:
    """Get Reddit credentials from environment."""
    return {
        "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "username": os.getenv("REDDIT_USERNAME", ""),
        "password": os.getenv("REDDIT_PASSWORD", ""),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "AWET/1.0"),
    }


def validate_credentials(creds: dict[str, str]) -> list[str]:
    """Validate credentials are present. Returns list of errors."""
    errors = []
    
    if not creds["client_id"]:
        errors.append("REDDIT_CLIENT_ID is missing")
    elif len(creds["client_id"]) < 10:
        errors.append(f"REDDIT_CLIENT_ID looks too short ({len(creds['client_id'])} chars, expected ~14)")
    
    if not creds["client_secret"]:
        errors.append("REDDIT_CLIENT_SECRET is missing")
    elif len(creds["client_secret"]) < 20:
        errors.append(f"REDDIT_CLIENT_SECRET looks too short ({len(creds['client_secret'])} chars, expected ~27)")
    
    if not creds["username"]:
        errors.append("REDDIT_USERNAME is missing")
    
    if not creds["password"]:
        errors.append("REDDIT_PASSWORD is missing")
    
    return errors


def get_access_token(creds: dict[str, str]) -> tuple[str | None, str]:
    """
    Get OAuth access token using password grant flow.
    
    Returns:
        (access_token, error_message) - token is None on failure
    """
    auth = requests.auth.HTTPBasicAuth(creds["client_id"], creds["client_secret"])
    
    data = {
        "grant_type": "password",
        "username": creds["username"],
        "password": creds["password"],
    }
    
    headers = {
        "User-Agent": creds["user_agent"],
    }
    
    try:
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            data=data,
            headers=headers,
            timeout=30,
        )
        
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"
        
        result = response.json()
        
        if "error" in result:
            return None, f"OAuth error: {result.get('error')} - {result.get('error_description', '')}"
        
        token = result.get("access_token")
        if not token:
            return None, f"No access_token in response: {result}"
        
        return token, ""
        
    except requests.RequestException as e:
        return None, f"Request failed: {e}"


def get_user_info(access_token: str, user_agent: str) -> tuple[dict | None, str]:
    """
    Get current user info from Reddit API.
    
    Returns:
        (user_info, error_message) - user_info is None on failure
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": user_agent,
    }
    
    try:
        response = requests.get(
            "https://oauth.reddit.com/api/v1/me",
            headers=headers,
            timeout=30,
        )
        
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"
        
        return response.json(), ""
        
    except requests.RequestException as e:
        return None, f"Request failed: {e}"


def main() -> int:
    print()
    print("=" * 60)
    print("REDDIT OAUTH SETUP HELPER")
    print("=" * 60)
    print()
    
    # Load .env
    load_dotenv()
    
    # Get credentials
    creds = get_credentials()
    
    print("Checking credentials...")
    print(f"  REDDIT_CLIENT_ID:     {'✓ set' if creds['client_id'] else '✗ missing'} ({len(creds['client_id'])} chars)")
    print(f"  REDDIT_CLIENT_SECRET: {'✓ set' if creds['client_secret'] else '✗ missing'} ({len(creds['client_secret'])} chars)")
    print(f"  REDDIT_USERNAME:      {'✓ set' if creds['username'] else '✗ missing'}")
    print(f"  REDDIT_PASSWORD:      {'✓ set' if creds['password'] else '✗ missing'}")
    print(f"  REDDIT_USER_AGENT:    {creds['user_agent']}")
    print()
    
    # Validate
    errors = validate_credentials(creds)
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"   - {error}")
        print()
        print("Add these to your .env file:")
        print()
        print("  REDDIT_CLIENT_ID=your_14_char_client_id")
        print("  REDDIT_CLIENT_SECRET=your_27_char_secret")
        print("  REDDIT_USERNAME=your_reddit_username")
        print("  REDDIT_PASSWORD=your_reddit_password")
        print("  REDDIT_USER_AGENT=AWET/1.0 (by /u/your_username)")
        print()
        print("Get credentials at: https://www.reddit.com/prefs/apps")
        return 1
    
    print("✓ All credentials present")
    print()
    
    # Test OAuth
    print("Testing OAuth authentication...")
    access_token, error = get_access_token(creds)
    
    if not access_token:
        print(f"❌ OAuth failed: {error}")
        print()
        print("Common issues:")
        print("  - Wrong client_id/secret: Double-check you copied them correctly")
        print("  - Wrong username/password: Check for typos, 2FA might block this")
        print("  - App type: Make sure you created a 'script' type app")
        return 1
    
    print("✓ OAuth token obtained")
    print()
    
    # Get user info
    print("Fetching user info...")
    user_info, error = get_user_info(access_token, creds["user_agent"])
    
    if not user_info:
        print(f"❌ API call failed: {error}")
        return 1
    
    print(f"✓ Authenticated as: u/{user_info.get('name', 'unknown')}")
    print(f"  Karma: {user_info.get('total_karma', 0):,}")
    print(f"  Account created: {user_info.get('created_utc', 'unknown')}")
    print()
    
    print("=" * 60)
    print("✅ SUCCESS! Reddit API credentials are working.")
    print("=" * 60)
    print()
    print("You can now run:")
    print("  python scripts/ingest_reddit_july_2025.py --write-db --update-features")
    print()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
