#!/bin/bash

venv='.work'
echo Install Virtualenv
which -s virtualenv || pip install virtualenv

# install python3
which -s $venv/bin/python3 || virtualenv -p python3 .work

# source
source $venv/bin/activate

# Install requirements
pip3 install -r requirements.txt --no-cache-dir
