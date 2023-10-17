%cd /content
from google.colab import drive
drive.mount('/content/drive')

import binascii
sdw = binascii.unhexlify("737461626c652d646966667573696f6e2d7765627569").decode('ascii')
w = binascii.unhexlify("73642d7765627569").decode('ascii')
webui_dir = f'/content/{sdw}'
gwebui_dir = f'/content/drive/MyDrive/{sdw}'
%env PYTHONDONTWRITEBYTECODE=1

#rsync sdwebui
!apt -y install -qq aria2
!apt --fix-broken install
#!apt -y install -qq aria2 git libcairo2-dev pkg-config python3-dev
#!git clone -q --branch master https://github.com/AUTOMATIC1111/$sdw && git -C {webui_dir} reset --hard && git -C {webui_dir} pull
!git clone -q --branch master https://github.com/AUTOMATIC1111/$sdw
!rsync -avqu {webui_dir} /content/drive/MyDrive/

#embeddings&positive&scripts
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d {gwebui_dir}/models/ESRGAN -o 4x-UltraSharp.pth
!wget https://raw.githubusercontent.com/camenduru/$sdw-scripts/main/run_n_times.py -O {gwebui_dir}/scripts/run_n_times.py

!rm -rf {webui_dir}/embeddings/negative && rm -rf {webui_dir}/models/Lora/positive
!git clone https://huggingface.co/embed/negative {webui_dir}/embeddings/negative
!rsync -avqu --exclude '.git' --exclude '.gitattributes' {webui_dir}/embeddings/negative {gwebui_dir}/embeddings
!git clone https://huggingface.co/embed/lora {webui_dir}/models/Lora/positive
!rsync -avqu --exclude '.git' --exclude '.gitattributes' {webui_dir}/models/Lora/positive {gwebui_dir}/models/Lora

#插件&config
if not os.path.exists('/content/extensions.py'):
  !wget https://raw.githubusercontent.com/Van-wise/sd-colab/main/on_drive/extensions.py
!python /content/extensions.py

#自定义插件 ***=插件名称 xxx=插件项目地址
#!rm -rf {gwebui_dir}/extensions/***
#!git clone xxx {gwebui_dir}/extensions/***
#!git -C {gwebui_dir}/extensions/*** reset --hard && git -C {gwebui_dir}/extensions/*** pull

#CKPT Model Download
#gmodel_dir = f"{gwebui_dir}/models/Stable-diffusion"
#!wget https://huggingface.co/ckpt/chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors -O {model_dir}/chilloutmix_NiPrunedFp32Fix.safetensors
#!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/dawn6666/real-max-v3.4/resolve/main/real-max-v3.4.safetensors -d {gmodel_dir} -o real-max-v3.4.safetensors

#CLIP
if not os.path.exists(f"{gwebui_dir}/models/CLIP"):
  os.mkdir(f"{gwebui_dir}/models/CLIP")
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt -d {gwebui_dir}/models/CLIP -o ViT-L-14.pt

%cd {gwebui_dir}
#!python launch.py --skip-torch-cuda-test

print("SDwebui on google drive.")
