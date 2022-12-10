import os
import sys

from launch import run, git_clone, repo_dir

name = "Smart Crop"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
print(f"loading {name} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -r "{req_file}"', f"Checking {name} requirements.",
    f"Couldn't install {name} requirements.")
blip_repo = "https://github.com/pharmapsychotic/BLIP"
git_clone(blip_repo, repo_dir('BLIP'), "BLIP")
