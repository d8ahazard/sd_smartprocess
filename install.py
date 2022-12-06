import os
import sys

from launch import run

name = "Smart Crop"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
print(f"loading {name} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -r "{req_file}"', f"Checking {name} requirements.", f"Couldn't install {name} requirements.")