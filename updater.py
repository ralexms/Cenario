#!/usr/bin/env python3
# updater.py — Checks for updates from GitHub

import os
import sys
import json
import subprocess
import urllib.request
import shutil
import zipfile
import tempfile

GITHUB_REPO = "ralexms/Cenario"
VERSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.txt')
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def get_current_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r') as f:
            return f.read().strip()
    return "0.0.0"

def get_latest_release():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error checking for updates: {e}")
        return None

def download_and_extract(url):
    print(f"Downloading update from {url}...")
    tmp_path = None
    try:
        # Download zip
        with urllib.request.urlopen(url) as response:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                shutil.copyfileobj(response, tmp_file)
                tmp_path = tmp_file.name
        
        print("Extracting update...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            
            # Find the inner folder (usually RepoName-TagName)
            extracted_items = os.listdir(tmp_dir)
            if not extracted_items:
                raise Exception("Empty zip file")
            
            # GitHub archives usually have a top-level folder
            source_dir = os.path.join(tmp_dir, extracted_items[0])
            if not os.path.isdir(source_dir):
                # If zip content is flat (unlikely for GitHub archives but possible)
                source_dir = tmp_dir

            print(f"Installing update to {APP_DIR}...")
            
            # Copy files to APP_DIR
            for item in os.listdir(source_dir):
                s = os.path.join(source_dir, item)
                d = os.path.join(APP_DIR, item)
                
                # Skip unnecessary files/folders
                if item in ['.git', '.github', '.gitignore', 'README.md', 'requirements.txt']:
                    continue
                
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return True

    except Exception as e:
        print(f"Update failed: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

def install_requirements():
    """Install pip dependencies from the deployed requirements file."""
    req_file = os.path.join(APP_DIR, 'installer', 'requirements-pip.txt')
    if not os.path.exists(req_file):
        return
    print("Installing/updating dependencies...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', req_file, '--quiet'],
            check=True,
        )
        print("Dependencies updated.")
    except Exception as e:
        print(f"Warning: dependency installation failed: {e}")

def update():
    print("Checking for updates...")
    current = get_current_version()
    release = get_latest_release()

    if not release:
        print("Could not determine latest version.")
        return

    latest = release['tag_name']
    
    if latest == current:
        print(f"You are already on the latest version ({current}).")
        return

    print(f"New version available: {latest} (current: {current})")
    choice = input("Do you want to update? [y/N]: ").strip().lower()
    if choice != 'y':
        return

    # Try to find zipball url
    zip_url = release.get('zipball_url')
    if not zip_url:
        # Fallback to source code zip
        zip_url = f"https://github.com/{GITHUB_REPO}/archive/refs/tags/{latest}.zip"

    if download_and_extract(zip_url):
        install_requirements()
        print("Update successful! Please restart the application.")
        with open(VERSION_FILE, 'w') as f:
            f.write(latest)
    else:
        print("Update failed.")

if __name__ == "__main__":
    update()
