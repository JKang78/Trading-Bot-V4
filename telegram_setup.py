"""
Helper to find your Telegram chat ID and test notifications.
Your trading bot does NOT reply to /start — this script does the setup for you.
"""

import os
import requests
from pathlib import Path


def load_env_file(path: str = ".env") -> dict:
    """Load KEY=value pairs from a .env file."""
    values = {}
    env_path = Path(path)
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def main():
    env = load_env_file()
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")

    print("=" * 50)
    print("TELEGRAM SETUP HELPER")
    print("=" * 50)

    if not token:
        print("\n❌ TELEGRAM_BOT_TOKEN is empty in .env")
        print("   1. Open @BotFather → /mybots → your bot → API Token")
        print("   2. Paste token into .env")
        print("   3. SAVE the file (Cmd+S)")
        return

    print(f"\n✓ Bot token found ({len(token)} chars)")

    # Check bot is valid
    me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10).json()
    if not me.get("ok"):
        print(f"\n❌ Invalid bot token: {me}")
        print("   Get a fresh token from @BotFather")
        return

    bot_name = me["result"].get("username", "?")
    print(f"✓ Bot is valid: @{bot_name}")

    print("\n--- Finding your Chat ID ---")
    print(f"1. Open Telegram and message @{bot_name}")
    print("2. Send: /start")
    print("3. Press Enter here after you've sent /start...")
    input()

    updates = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates",
        timeout=10,
    ).json()

    if not updates.get("ok"):
        print(f"❌ Error: {updates}")
        return

    messages = updates.get("result", [])
    if not messages:
        print("\n❌ No messages found.")
        print("   Make sure you sent /start to @" + bot_name)
        print("   NOT to @userinfobot or another bot.")
        return

    # Get most recent user who messaged the bot
    last = messages[-1]
    from_user = last.get("message", {}).get("from", {})
    found_chat_id = last.get("message", {}).get("chat", {}).get("id")

    print(f"\n✅ Your Chat ID is: {found_chat_id}")
    print(f"   Name: {from_user.get('first_name', '')} (@{from_user.get('username', 'no username')})")
    print(f"\nAdd this to your .env:")
    print(f"TELEGRAM_CHAT_ID={found_chat_id}")

    # Test send
    test_id = chat_id or str(found_chat_id)
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": test_id, "text": "✅ Kraken trading bot is connected!"},
        timeout=10,
    )
    if r.json().get("ok"):
        print("\n✅ Test message sent! Check Telegram.")
    else:
        print(f"\n⚠️ Test send failed: {r.text}")


if __name__ == "__main__":
    main()
