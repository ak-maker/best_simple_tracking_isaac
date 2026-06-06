from setuptools import setup, find_packages

setup(
    name="best_simple_tracking",
    version="0.1.0",
    description="Model-based RL agent for active multi-target tracking, ported to Isaac Lab.",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "best_simple_tracking": [
            "assets/*.usd",
            "assets/*.urdf",
            "assets/*.pt",
            "params/*.yaml",
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        # Isaac Lab + Isaac Sim must be installed separately (see README).
        # Only the runtime libs not already pulled in by Isaac Lab.
        "numpy",
        "pyyaml",
        "tensorboard",
        "psutil",
    ],
)
