#!/usr/bin/env python3
"""
Verify Alpaca environment configuration.

This script checks that Alpaca credentials are properly configured
without printing the actual secrets.

Usage:
    python scripts/verify_alpaca_env.py
    make verify-alpaca
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


def mask_secret(value: str | None, show_chars: int = 4) -> str:
    """Mask a secret, showing only first few characters."""
    if not value:
        return "❌ NOT SET"
    if len(value) <= show_chars:
        return "****"
    return value[:show_chars] + "*" * (len(value) - show_chars)


def verify_alpaca_env() -> bool:
    """
    Verify Alpaca environment variables are configured.
    
    Returns:
        True if all required vars are set, False otherwise
    """
    # Load .env from repo root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env from: {env_path}")
    else:
        print(f"⚠️  No .env file found at: {env_path}")
    
    # Check required variables
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    print("\n" + "=" * 50)
    print("🦙 Alpaca Environment Configuration")
    print("=" * 50)
    
    # Display configuration (secrets masked)
    print(f"\nALPACA_API_KEY:    {mask_secret(api_key)}")
    print(f"ALPACA_SECRET_KEY: {mask_secret(secret_key)}")
    print(f"ALPACA_BASE_URL:   {base_url}")
    
    # Validate
    errors = []
    
    if not api_key:
        errors.append("ALPACA_API_KEY is not set")
    
    if not secret_key:
        errors.append("ALPACA_SECRET_KEY is not set")
    
    if base_url and "paper-api" not in base_url:
        errors.append(f"ALPACA_BASE_URL is not paper endpoint: {base_url}")
    
    # Print results
    print("\n" + "-" * 50)
    
    if errors:
        print("❌ Configuration INVALID:\n")
        for error in errors:
            print(f"   • {error}")
        print("\n💡 Fix: Add missing variables to .env file:")
        print("   ALPACA_API_KEY=PK...")
        print("   ALPACA_SECRET_KEY=...")
        print("   ALPACA_BASE_URL=https://paper-api.alpaca.markets")
        return False
    
    print("✅ Alpaca configuration is VALID")
    print("   • API key is set")
    print("   • Secret key is set")
    print("   • Using paper trading endpoint")
    
    # Try to import and validate the client
    try:
        from src.integrations.alpaca_client import AlpacaClient
        client = AlpacaClient.from_env()
        print(f"   • AlpacaClient initialized successfully")
        print(f"   • Base URL: {client.base_url}")
    except Exception as e:
        print(f"\n⚠️  AlpacaClient initialization failed: {e}")
        return False
    
    return True


def main() -> int:
    """Main entry point."""
    success = verify_alpaca_env()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
