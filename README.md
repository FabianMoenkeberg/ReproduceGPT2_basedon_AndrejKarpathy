# ReproduceGPT2_basedon_AndrejKarpathy
Goal is to obtain detailed understanding of the GPT-model by reproducing the GPT-2 model following the video of  [Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU).

I know alread a lot about the theory of transformer models, including some [finetuning of Segformer](https://github.com/FabianMoenkeberg/SegFormer) using LoRA. However, a lot of practical details are still unclear to me. Especially, when it comes to training  a network like this.
By following the guide of Andrej Karpathy I hope to fill those gaps and abtain a clear picture about the complete process.

A friend and former colleague from EPFL, Luca Pegolotti, inspired me with his [article](https://medium.com/@pegolotti.luca/lets-reproduce-gpt-2-again-368711e0d1c5).

# Code Reference
The code from Andrej can be found [here](https://github.com/karpathy/build-nanogpt).

The code base of Luca can be found [here](https://github.com/lucapegolotti/gpt-2).


# Installation and Setup
## With Docker and VS Code
Important steps, some are already done, e.g. tensorflow[and-cuda]==2.17.0
* Install WSL 2 and VS Code
* Install Cuda-Driver ???
* Use tensorflow[and-cuda]==2.17.0 instead of tensorflow or tensorflow-gpu
* General Infos [Cuda and Docker ](https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html)
* Install [cuda-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)
* Set Paths in `~/.bashrc` 
  `export CUDA_PATH=$CUDA_PATH:/usr/local/cuda/`
  `export PATH=$PATH:$CUDA_PATH`
  `export PATH=$PATH:/usr/local/cuda/bin`
  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib`
  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu`
* Install [cuda-driver]()
* Install [Cudnn-cuda-12](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) or cudnn-cuda-11 depending on coda version
  Maybe create link from to `/usr/local/cuda`
  Set path in `~/.bashrc` :  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu`
* Start Docker `sudo service docker start`
* Create Docker container from Dockerfile `docker buildx build -t gpt2_image .`
* Run Docker container once `docker run --gpus all --name gpt2_training -p 2224:24  -it -v /mnt/c/Users/fmo/sourcePrivate/ReproduceGPT2_basedon_AndrejKarpathy:/app gpt2_image tail -f /dev/null` (not debugging)
* Debug Docker container ``*
* Inside the file [/etc/nvidia-container-runtime/config.toml change no-cgroups from true to false](https://github.com/microsoft/WSL/issues/9962#issuecomment-2066459177)
  
### Definitions to use Docker in VS Code
Important files are 
* `Dockerfile`
* `.vscode/launch.json` -> Launch tasks that can be run directly with `Play`-Button 
* `.vscode/tasks.json` -> Definition of single tasks, e.g., build, run (command, file, shell)
* `.devcontainer/devcontainer.json` (this might be not used or wrong)

## Run and Debug with VS Code
Note: It is important to use `buildx` not `build`.

### Start Docker Container and attach VS Code to running container.
* Build manually docker container in terminal: `docker buildx build -t gpt2_image .`
* Start Container in the Container-Field.
* Right-Click on running container and Attach VS Code to running container.
* Debug and run scripts with launch-json "Python: Run current File inside Docker".

### Run it in VS Code manually
* Build manually docker container in terminal: `docker buildx build -t gpt2_image .`
* Comment "Wait for client from debugpy"
* Run it manually in terminal: `docker run --gpus all -v /mnt/c/Users/fmo/sourcePrivate/ReproduceGPT2_basedon_AndrejKarpathy:/app gpt2_image`

### Debug it with VS Code

* Build manually docker container in terminal: `docker buildx build -t ## With Docker and VS Code
Important steps, some are already done, e.g. tensorflow[and-cuda]==2.17.0
* Install WSL 2 and VS Code
* Install Cuda-Driver ???
* Use tensorflow[and-cuda]==2.17.0 instead of tensorflow or tensorflow-gpu
* General Infos [Cuda and Docker ](https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html)
* Install [cuda-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)
* Set Paths in `~/.bashrc` 
  `export CUDA_PATH=$CUDA_PATH:/usr/local/cuda/`
  `export PATH=$PATH:$CUDA_PATH`
  `export PATH=$PATH:/usr/local/cuda/bin`
  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib`
  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu`
* Install [cuda-driver]()
* Install [Cudnn-cuda-12](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) or cudnn-cuda-11 depending on coda version
  Maybe create link from to `/usr/local/cuda`
  Set path in `~/.bashrc` :  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu`
* Start Docker `sudo service docker start`
* Create Docker container from Dockerfile `docker buildx build -t gpt2_image .`
* Run Docker container once `docker run --gpus all --name gpt2_training -p 2224:24  -it -v /mnt/c/Users/fmo/sourcePrivate/ReproduceGPT2_basedon_AndrejKarpathy:/app gpt2_image tail -f /dev/null` (not debugging)
* Debug Docker container ``*
* Inside the file [/etc/nvidia-container-runtime/config.toml change no-cgroups from true to false](https://github.com/microsoft/WSL/issues/9962#issuecomment-2066459177)
  
### Definitions to use Docker in VS Code
Important files are 
* `Dockerfile`
* `.vscode/launch.json` -> Launch tasks that can be run directly with `Play`-Button 
* `.vscode/tasks.json` -> Definition of single tasks, e.g., build, run (command, file, shell)
* `.devcontainer/devcontainer.json` (this might be not used or wrong)

# Run and Debug with VS Code
Note: It is important to use `buildx` not `build`.

## Start Docker Container and attach VS Code to running container.
* Build manually docker container in terminal: `docker buildx build -t gpt2_image .`
* Start Container in the Container-Field.
* Right-Click on running container and Attach VS Code to running container.
* Debug and run scripts with launch-json "Python: Run current File inside Docker".

## Run it in VS Code manually
* Build manually docker container in terminal: `docker buildx build -t ## With Docker and VS Code
Important steps, some are already done, e.g. tensorflow[and-cuda]==2.17.0
* Install WSL 2 and VS Code
* Install Cuda-Driver ???
* Use tensorflow[and-cuda]==2.17.0 instead of tensorflow or tensorflow-gpu
* General Infos [Cuda and Docker ](https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html)
* Install [cuda-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)
* Set Paths in `~/.bashrc` 
  `export CUDA_PATH=$CUDA_PATH:/usr/local/cuda/`
  `export PATH=$PATH:$CUDA_PATH`
  `export PATH=$PATH:/usr/local/cuda/bin`
  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib`
  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu`
* Install [cuda-driver]()
* Install [Cudnn-cuda-12](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) or cudnn-cuda-11 depending on coda version
  Maybe create link from to `/usr/local/cuda`
  Set path in `~/.bashrc` :  `LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu`
* Start Docker `sudo service docker start`
* Create Docker container from Dockerfile `docker buildx build -t gpt2_image .`
* Run Docker container once `docker run --gpus all --name gpt2_training -p 2224:25  -it -v /mnt/c/Users/fmo/sourcePrivate/ReproduceGPT2_basedon_AndrejKarpathy:/app -v /mnt/d/data:/data gpt2_image tail -f /dev/null` (not debugging)
* Debug Docker container ``*
* Inside the file [/etc/nvidia-container-runtime/config.toml change no-cgroups from true to false](https://github.com/microsoft/WSL/issues/9962#issuecomment-2066459177)
  
### Definitions to use Docker in VS Code
Important files are 
* `Dockerfile`
* `.vscode/launch.json` -> Launch tasks that can be run directly with `Play`-Button 
* `.vscode/tasks.json` -> Definition of single tasks, e.g., build, run (command, file, shell)
* `.devcontainer/devcontainer.json` (this might be not used or wrong)

# Run and Debug with VS Code
Note: It is important to use `buildx` not `build`.

## Start Docker Container and attach VS Code to running container.
* Build manually docker container in terminal: `docker buildx build -t gpt2_image .`
* Start Container in the Container-Field.
* Right-Click on running container and Attach VS Code to running container.
* Debug and run scripts with launch-json "Python: Run current File inside Docker".

## Run it in VS Code manually
* Build manually docker container in terminal: `docker buildx build -t gpt2_image .`
* Comment "Wait for client from debugpy"
* Run it manually in terminal: `docker run --gpus all -v /mnt/c/Users/fmo/sourcePrivate/ReproduceGPT2_basedon_AndrejKarpathy:/app -v /mnt/d/data:/data gpt2_image`

## Debug it with VS Code

* Build manually docker container in terminal: `docker buildx build -t gpt2_image .`
* Uncomment "Wait for client from debugpy"
* Start "Terminal -> Run Task..." the task "docker-run: prebuild"
* In "Run and Debug" run "Docker: Python - Attach Debug"
* debug it ...*
 .`
* Comment "Wait for client from debugpy"
* Run it manually in terminal: `docker run --gpus all -v /mnt/c/Users/fmo/sourcePrivate/ReproduceGPT2_basedon_AndrejKarpathy:/app -v /mnt/d/data:/data gpt2_image`

## Debug it with VS Code

* Build manually docker container in terminal: `docker buildx build -t gpt2_image .`
* Uncomment "Wait for client from debugpy"
* Start "Terminal -> Run Task..." the task "docker-run: prebuild"
* In "Run and Debug" run "Docker: Python - Attach Debug"
* debug it ...*
 .`
* Uncomment "Wait for client from debugpy"
* Start "Terminal -> Run Task..." the task "docker-run: prebuild"
* In "Run and Debug" run "Docker: Python - Attach Debug"
* debug it ...*
