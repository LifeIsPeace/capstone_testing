# Python Version
I shouldddd do python 3.9 if dependencies allow it
python = 3.10

# Links for General Set up
To preview markdown files without downloading any extensions.</br>
https://stackoverflow.com/questions/73049432/how-can-i-open-a-md-file-in-vs-code-preview-mode-from-terminal

How can we use pyproject.toml with conda?
https://stackoverflow.com/questions/76722680/what-is-the-best-way-to-combine-conda-with-standard-python-packaging-tools-e-g

# Notes about command line
Argsparse is the command line tool used for python (Ex: pip install -r requirements.txt | What does "-r" mean in this context. That's argsparse)</br>
https://docs.python.org/3/library/argparse.html

# Notes about pyproject.toml
Must read for implementation <br/>
https://pydevtools.com/handbook/explanation/pyproject-vs-requirements/

# Notes that may or may not be used
Requirement.txt files <br/>
https://pip.pypa.io/en/stable/user_guide/#requirements-files

pyproject.toml dependencies/requirement equivalent (requirements.txt )explanation <br/>
https://stackoverflow.com/questions/62408719/download-dependencies-declared-in-pyproject-toml-using-pip

So requrement.txt files are for concrete dependencies; meaning exactly which package and what version. Ex: python == 3.10

pyproject.toml is for abtract dependencies; meaning it uses version specifiers to allow for different versions to be used. Ex: python > 3.9