#!/usr/bin/env python

import os
import shutil
import stat

from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

from mindface.version import __version__

version = __version__
package_name = 'mindface'
cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, 'build')


def clean():
    # pylint: disable=unused-argument
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(os.path.join(cur_dir, 'build')):
        shutil.rmtree(os.path.join(cur_dir, 'build'), onerror=readonly_handler)
    if os.path.exists(os.path.join(cur_dir, f'{package_name}.egg-info')):
        shutil.rmtree(os.path.join(cur_dir, f'{package_name}.egg-info'), onerror=readonly_handler)


clean()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


class EggInfo(egg_info):
    """Egg info."""

    def run(self):
        super().run()
        egg_info_dir = os.path.join(cur_dir, f'{package_name}.egg-info')
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""

    def run(self):
        super().run()
        mindarmour_dir = os.path.join(pkg_dir, 'lib', package_name)
        update_permissions(mindarmour_dir)


setup(
    name=package_name,
    version=version,
    author="MindLab-AI",
    url="https://github.com/mindlab-ai/mindface",
    project_urls={
        'Sources': 'https://github.com/mindlab-ai/mindface',
        'Issue Tracker': 'https://github.com/mindlab-ai/mindface/issues',
    },
    description="An open source computer vision research tool box.",
    license='Apache 2.0',
    include_package_data=True,
    packages=find_packages(exclude=("mindface")),
    cmdclass={
        'egg_info': EggInfo,
        'build_py': BuildPy,
    },
    install_requires=[
        'mindspore_gpu==1.8.0',
        'numpy==1.21.6',
        'opencv_python==4.6.0.66',
        'scipy==1.7.3',
        'pyyaml>=5.3',
        "scikit-learn==1.1.2",
        "Pillow==9.2.0",
        "matplotlib==3.6.0"
    ]
)
print(find_packages())
