"""Install script for setuptools."""

from setuptools import setup, find_packages

# The flybody package can be installed in three modes:
#
# 1. Core installation: light-weight installation for experimenting with the
#    fly model in MuJoCo or with dm_control task environments. ML components
#    such as Tensorflow and Acme are not installed and policy rollouts and
#    training are not supported.
#    To install, use: pip install -e .
#
# 2. Add ML components: same as (1), plus Tensorflow, Acme to allow bringing
#    policy networks into play (e.g. for inference), but without training them.
#    To install, use: pip install -e .[tf]
#
# 3. Add training components: Same as (1) and (2), plus Ray to also allow
#    distributed policy training in the dm_control task environments.
#    To install, use: pip install -e .[ray]

core_requirements = [
    "numpy==1.26.4",
    "dm_control",
    "h5py",
    "pytest",
    "mediapy",
]

tf_requirements = [
    "dm-acme[tf,envs,jax]",
    "tensorflow==2.8.0",
    "tensorflow-probability==0.16.0",
    "dm-reverb==0.7.0",
]

ray_requirements = tf_requirements + [
    "ray[default]",
]

dev_requirements = [
    "yapf",
    "ruff",
    "jupyterlab",
    "tqdm",
]

setup(
    name="vnl_ray",
    version="0.1",
    packages=find_packages(),
    package_data={
        "vnl-ray": ["vnl-ray/assets/*.obj", "vnl-ray/assets/*.xml"],
    },
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "tf": tf_requirements,
        "ray": ray_requirements,
        "dev": dev_requirements,
    },
    author="Roman Vaxenburg, Yuval Tassa, Zinovia Stefanidi, Scott Yang, Eric Leonardis",
    description="VNL training with ray backend, forked from the flybody repo from Turaga Lab",
    url="https://github.com/talmolab/vnl-ray",
)
