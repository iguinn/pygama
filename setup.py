 #!/usr/bin/env python3
import os
import re
import sys
import sysconfig
import platform
import subprocess
import glob
import numpy

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.egg_info import egg_info
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as TestCommand
from shutil import copyfile, copymode

class PygamaExt(build_ext):
    def run(self):
        # update the submodule
        print("Updating git submodules...")
        import git
        repo = git.Repo(os.path.dirname(os.path.realpath(__file__)))
        repo.git.submodule('update', '--init', '--recursive', '--depth=1')

        build_ext.run(self)

def make_git_file():
    print("Creating pygama/git.py")
    try:
        import git
        repo = git.Repo(os.path.dirname(os.path.realpath(__file__)))
        with open(repo.working_tree_dir + '/pygama/git.py', 'w') as f:
            f.write("branch = '" + repo.git.describe('--all') + "'\n")
            f.write("revision = '" + repo.head.commit.hexsha +"'\n")
            f.write("commit_date = '" + str(repo.head.commit.committed_datetime) + "'\n")
    except Exception as ex:
        print(ex)
        print('continuing...')

#Add a git hook to clean jupyter notebooks before commiting
def clean_jupyter_notebooks():
    import git
    repo = git.Repo(os.path.dirname(os.path.realpath(__file__)))
    with repo.config_writer('repository') as config:
        try:
            import nbconvert
            if nbconvert.__version__[0] < '6': #clear output
                fil=""" "jupyter nbconvert --stdin --stdout --log-level=ERROR\\
                --to notebook --ClearOutputPreprocessor.enabled=True" """
            else: # also clear metadata
                fil=""" "jupyter nbconvert --stdin --stdout --log-level=ERROR\\
                --to notebook --ClearOutputPreprocessor.enabled=True\\
                --ClearMetadataPreprocessor.enabled=True" """
        except:
            # if nbconvert (part of jupyter) is not installed, disable filter
            fil = "cat"

        config.set_value('filter "jupyter_clear_output"', 'clean', fil)
        config.set_value('filter "jupyter_clear_output"', 'smudge', 'cat')
        config.set_value('filter "jupyter_clear_output"', 'required', 'false')


# run during installation; this is when files get copied to build dir
class PygamaBuild(build_py):
    def run(self):
        make_git_file()
        clean_jupyter_notebooks()
        build_py.run(self)

# run during local installation; in this case build_py isn't run...
class PygamaDev(develop):
    def run(self):
        make_git_file()
        clean_jupyter_notebooks()
        develop.run(self)

setup(
    name='pygama',
    version='0.5',
    author='LEGEND',
    author_email='wisecg@uw.edu',
    description='Python package for decoding and processing digitizer data',
    long_description='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scimath',
        'numba',
        'parse',
        'GitPython',
        'tinydb',
        'pyFFTW',
        'h5py>=3.2.0',
        'pandas',
        'matplotlib'
        # 'fcutils @ https://github.com/legend-exp/pyfcutils.git#egg=1.0.0'
    ],
    ext_modules=[Extension(name = 'pygama.dsp.eigama.' + os.path.splitext(os.path.basename(src))[0],
                           sources = [src],
                           include_dirs=[
                               'pygama/dsp/eigama',
                               'pygama/dsp/eigama/eigen',
                               numpy.get_include()
                           ],
                           extra_compile_args=["-std=c++17"]
                           )
                 for src in glob.glob('pygama/dsp/eigama/*.cpp')],
    cmdclass=dict(build_ext=PygamaExt, build_py=PygamaBuild, develop=PygamaDev),
    zip_safe=False,
)
