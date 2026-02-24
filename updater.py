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

def restart_application():
    """Restart the application."""
    print("Restarting application...")
    python = sys.executable
    script = os.path.join(APP_DIR, 'cenario.py')
    
    if sys.platform == 'win32':
        # Windows restart
        subprocess.Popen([python, script], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # Unix restart
        os.execl(python, python, script)

def check_for_updates():
    """Check for updates and return metadata."""
    current = get_current_version()
    release = get_latest_release()

    if not release:
        return {'error': 'Could not check for updates'}

    latest = release['tag_name']
    if latest == current:
        return {'update_available': False, 'current_version': current}

    # Try to find zipball url
    zip_url = release.get('zipball_url')
    if not zip_url:
        zip_url = f"https://github.com/{GITHUB_REPO}/archive/refs/tags/{latest}.zip"

    return {
        'update_available': True,
        'current_version': current,
        'latest_version': latest,
        'release_notes': release.get('body', 'No release notes available.'),
        'download_url': zip_url
    }

def perform_update(url, version):
    """Download and install the update."""
    if download_and_extract(url):
        install_requirements()
        with open(VERSION_FILE, 'w') as f:
            f.write(version)
        return True
    return False

if __name__ == "__main__":
    # CLI usage
    res = check_for_updates()
    if res.get('error'):
        print(res['error'])
    elif res['update_available']:
        print(f"Update available: {res['latest_version']}")
        if input("Update? [y/N] ").lower() == 'y':
            if perform_update(res['download_url'], res['latest_version']):
                print("Updated. Restarting...")
                restart_application()
            else:
                print("Update failed.")
    else:
        print("Up to date.")
