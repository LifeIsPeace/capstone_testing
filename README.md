# Packages to note

-Cpu is default. You have to manually add gpu installation
-y accepts all terms and conditions etc

### This command PROBABLY installs gpu support which I don't need rn
conda create --name capstoneExperimentation python=3.10 pandas pytorch torchvision torchaudio -c conda-forge -y 

Here's the plan. Pip install everything pytorch related, and if it goes wrong THEN try conda. 

If this works add "nomkl" to conda install line

conda create --name capstoneEnv python=3.10 

Note that I am on Windows rn
DO NOT pip3 install sounddevice and librosa. That gave me issues

```
python -m pip install sounddevice
pip install librosa
pip install pandas
pip install pretty-midi
pip3 install torch torchvision
pip3 install torchaudio
pip install soundfile
```

## Python Version
Latest stable Pytorch version requires Python 3.10.
python = 3.10

## Windows BS that you might have to go through

- In conda if name isn't found, copy and paste the whole file name
- Never use SSH for github. Never works until it does. Trash

## Notes for General Set up

!!!! DO NOT FORGET GENERICS WHEN NEEDED (Remember Computer Networking)

Conda should install as many packages as it can on its own (at once/in one command), only then should pip be used. If more conda packages are needed after already using pip then it's best to remake the conda environment.

So an environment.yaml should be made for conda. And, if necessary, a requirements.txt should be made for the python packages (Yes you can add wildcard characters for the package's versions).

sounddevice must be installed with pip. Same with librosa

## README

"ctrl + shift + v" on windows to preview markdown

## Links for General Set up

### For sounddevice (causes so many issues)

https://github.com/pyinstaller/pyinstaller/issues/7065

**To preview markdown files without downloading any extensions.**</br>
https://stackoverflow.com/questions/73049432/how-can-i-open-a-md-file-in-vs-code-preview-mode-from-terminal

**!!! How can we use pyproject.toml with conda?**
https://stackoverflow.com/questions/76722680/what-is-the-best-way-to-combine-conda-with-standard-python-packaging-tools-e-g

## Notes about command line

**Argsparse is the command line tool used for python (Ex: pip install -r requirements.txt | What does "-r" mean in this context. That's argsparse)**</br>
https://docs.python.org/3/library/argparse.html

## Notes about pyproject.toml

So pyproject.toml in this context should be used for metadata and equivalently to requirements.txt. It really shines when using python package and project managers (handles virtual environments), but we're already using conda.

**Must read for implementation** <br/>
- https://pydevtools.com/handbook/explanation/pyproject-vs-requirements/

**About versioning**
- https://packaging.python.org/en/latest/discussions/versioning/

# Notes that may or may not be used
**Requirement.txt files** <br/>
https://pip.pypa.io/en/stable/user_guide/#requirements-files

**pyproject.toml dependencies/requirement equivalent (requirements.txt )explanation** <br/>
https://stackoverflow.com/questions/62408719/download-dependencies-declared-in-pyproject-toml-using-pip

So requrement.txt files are for concrete dependencies; meaning exactly which package and what version. Ex: python == 3.10

pyproject.toml is for abtract dependencies; meaning it uses version specifiers to allow for different versions to be used. Ex: python > 3.9

**Conda vs Pip** <br/>
https://www.anaconda.com/blog/understanding-conda-and-pip

**For uv (ultraviolet/better pip)**<br/>
I advise against using uv because it creates venv environments and adds uneeded complexity if you're already using conda. Conda is a MUST for this project.
- https://pydevtools.com/handbook/reference/uv/
- https://medium.com/@datagumshoe/using-uv-and-conda-together-effectively-a-fast-flexible-workflow-d046aff622f0