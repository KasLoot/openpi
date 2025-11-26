import sys
from pathlib import Path

# Add the src directory to the path so we can import openpi.
# This is needed because the notebook is in the examples directory.
sys.path.append(str(Path.cwd().parent / "src"))

import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.shared import download


config = _config.get_config("pi05_droid")
checkpoint_dir = Path("/media/yuxin/StorageDisk/Ubunutu_Files/models/pi/pi05_droid")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = droid_policy.make_droid_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)