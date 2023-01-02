#!.bin/bash

echo ">> Updating conda base..."
eval "conda update -n base conda -y"

env_name=$1
if [ -z "$env_name" ]
    then
        env_name="latest_env"
fi

echo ">> Creating conda environment with name '$env_name'..."
eval "conda env create -n $env_name -f environment_eva.yml"

echo ">> Making widgets work..."
eval "conda install -n base -c conda-forge jupyterlab_widgets -y"
eval "conda install -n $env_name -c conda-forge ipywidgets -y"
