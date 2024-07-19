import os
import sys

from launch import run, git_clone, repo_dir

name = "Smart Crop"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
print(f"loading {name} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -r "{req_file}"', f"Checking {name} requirements.",
    f"Couldn't install {name} requirements.")
print("Cloning BLIP repo...")
blip_repo = "https://github.com/pharmapsychotic/BLIP"
git_clone(blip_repo, repo_dir('BLIP'), "BLIP")

# git clone https://github.com/X-PLUG/mPLUG-Owl.git
# cd mPLUG-Owl/mPLUG-Owl2
# mplug_repo = "https://github.com/X-PLUG/mPLUG-Owl.git"
# mplug_owl_path = repo_dir('mplug_owl_src')
# git_clone(mplug_repo, mplug_owl_path, "mPLUG-Owl")
# mplug_owl2_sub1_path = os.path.join(mplug_owl_path, "mPLUG-Owl", "mplug_owl")
# mplug_owl2_sub2_path = os.path.join(mplug_owl_path, "mPLUG-Owl2", "mplug_owl2")
#
# sys.path.append(mplug_owl2_sub1_path)
# sys.path.append(mplug_owl2_sub2_path)
#
