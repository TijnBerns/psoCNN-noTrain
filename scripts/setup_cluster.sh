#! /usr/bin/env bash
set -e

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# install the `virtualenv` command
python3 -m pip install --upgrade pip
python3 -m pip install --user virtualenv

DATA_DIR=/scratch/"$USER"/data

# set up a virtual environment located at
# /scratch/$USER/virtual_environments/psocnn
# and make a symlink to the virtual environment
# at the root directory of this project called "venv"
echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN99 ###"
./setup_venv.sh

# Download the data 
source "$SCRIPT_DIR"/../venv/bin/activate
python3 "$SCRIPT_DIR"/download_data.py --root "$DATA_DIR"
deactivate

# make sure that there's also a virtual environment
# on the GPU nodes
echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
ssh cn47 "
  source .profile
  cd $PWD;
  ./setup_venv.sh
  source "$SCRIPT_DIR"/../venv/bin/activate
  python3 "$SCRIPT_DIR"/download_data.py --root "$DATA_DIR"
  deactivate

"

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN48 ###"
ssh cn48 "
  source .profile
  cd $PWD;
  ./setup_venv.sh
  source "$SCRIPT_DIR"/../venv/bin/activate
  python3 "$SCRIPT_DIR"/download_data.py --root "$DATA_DIR"
  deactivate
"

# make a symlink to the data in order to directly access it from the root of the project
mkdir -p "$SCRIPT_DIR"/../data
ln -sfn /scratch/tberns/data/ "$SCRIPT_DIR"/../data