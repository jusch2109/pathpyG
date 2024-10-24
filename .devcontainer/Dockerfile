# Hier von cudnn8 auf cudnn9 
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /workspaces/pathpyG
RUN apt-get update
RUN apt-get -y install git

# For signed commits: https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials#_sharing-gpg-keys
RUN apt install gnupg2 -y

# Install torch
# RUN pip install torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# CPU only
RUN pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu 

# Install torch_geometric and dependencies
RUN pip install torch_geometric>=2.5.0
# RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
# CPU only
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html 
