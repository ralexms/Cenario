#!/usr/bin/env python3
# updater.py — Checks for updates from GitHub

import os
import sys
import json
import urllib.request
import subprocess

GITHUB_REPO = "ralexms/Cenario"
VERSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.txt')

def get_current_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r') as f:
            return f.read().strip()
    return "0.0.0"

def get_latest_version():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            return data['tag_name']
    except Exception as e:
        print(f"Error checking for updates: {e}")
        return None

def update():
    print("Checking for updates...")
    current = get_current_version()
    latest = get_latest_version()

    if not latest:
        print("Could not determine latest version.")
        return

    if latest == current:
        print(f"You are already on the latest version ({current}).")
        return

    print(f"New version available: {latest} (current: {current})")
    choice = input("Do you want to update? [y/N]: ").strip().lower()
    if choice != 'y':
        return

    print("Updating...")
    try:
        subprocess.check_call(["git", "pull"])
        print("Update successful! Please restart the application.")
        with open(VERSION_FILE, 'w') as f:
            f.write(latest)
    except subprocess.CalledProcessError as e:
        print(f"Update failed: {e}")

if __name__ == "__main__":
    update()
