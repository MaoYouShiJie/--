#@title 👇 1.启动SDwebui！
#此单元格，每次连接colab都需要运行一次；作用是，安装运行环境并启动SDwebui。
import os
import sys
import glob
from IPython.utils import capture
from IPython.display import clear_output

import binascii
sdw = binascii.unhexlify("737461626c652d646966667573696f6e2d7765627569").decode('ascii')
w = binascii.unhexlify("73642d7765627569").decode('ascii')
webui_dir = f'/content/{sdw}'
gwebui_dir = f'/content/drive/MyDrive/{sdw}'

#Check GPU  ,不使用GPU会出现莫名其妙的错误哦。
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
if tf.test.gpu_device_name():
    print("GPU is available")
else:
    print("GPU is NOT available")
    raise Exception("\n没有使用GPU，请在代码执行程序-更改运行时类型-设置为GPU！\n如果不能使用GPU，建议更换账号！")

if not os.path.exists('/content/drive'):
  from google.colab import drive
  drive.mount('/content/drive')

!apt -y install -qq aria2
![ -d "{webui_dir}" ] || git clone -q https://github.com/AUTOMATIC1111/$sdw {webui_dir}

# Requirements stable-diffusion-stability-ai
with capture.capture_output() as cap:
  %cd /content
  #!rm -rf {gwebui_dir}/repositories/stable-diffusion-stability-ai
  !mkdir -p {gwebui_dir}/repositories/stable-diffusion-stability-ai
  !wget -q -i https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dependencies/A1111.txt
  !dpkg -i *.deb
  !tar -C /content/ --zstd -xf sd_mrep.tar.zst
  !tar -C / --zstd -xf gcolabdeps.tar.zst
  !rm *.deb | rm *.zst | rm *.txt
  if not os.path.exists(f'{gwebui_dir}/libtcmalloc/libtcmalloc_minimal.so.4'):
    !wget https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O {gwebui_dir}/libtcmalloc/libtcmalloc_minimal.so.4
    %env LD_PRELOAD={gwebui_dir}/libtcmalloc/libtcmalloc_minimal.so.4
  else:
    %env LD_PRELOAD={gwebui_dir}/libtcmalloc/libtcmalloc_minimal.so.4

  !rsync -avqu /content/sd/stablediffusion/* {gwebui_dir}/repositories/stable-diffusion-stability-ai && rm -rf /content/sd

!mkdir -p {webui_dir}/cache
os.environ['TRANSFORMERS_CACHE']=f"{webui_dir}/cache"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
!sed -i 's@text = _formatwarnmsg(msg)@text =\"\"@g' /usr/lib/python3.10/warnings.py
!apt --fix-broken install
!pip install open-clip-torch==2.20.0 -qq --no-deps
!pip install fastapi==0.94.0 gradio==3.41.2 -qq
%env PYTHONDONTWRITEBYTECODE=1
%env TF_CPP_MIN_LOG_LEVEL=1

#链接 controlnet
#@markdown 选择ControlNet：
#@markdown
#@markdown 0：不需要使用ControlNet
#@markdown
#@markdown 1：下载ControlNet 模型到 Colab
#@markdown
#@markdown 2：下载ControlNet 模型到 Google drive    (会占用云盘,不推荐！)
%cd /content
controlnet = 2 #@param {type:"slider", min:0, max:2, step:1}
cn_dir = f"{webui_dir}/extensions/{w}-controlnet/models"
gcn_dir = f"{gwebui_dir}/extensions/{w}-controlnet/models"
!mkdir -p {cn_dir} && mkdir -p {gcn_dir}
!test -d {cn_dir} && test -d {gcn_dir} && ln -sf {gcn_dir} {cn_dir}
!pip install -qq pycloudflared ngrok
!pip install -qq translators chardet openai boto3 aliyun-python-sdk-core aliyun-python-sdk-alimt python-dotenv

if controlnet == 1:
    # 下载ControlNet 模型到 Colab
    !wget -q -O cnmodels.py https://raw.githubusercontent.com/Van-wise/sd-colab/main/cnmodels.py
    from cnmodels import download
    from cnmodels import cndown_colab
    cndown_colab()
elif controlnet == 2:
    # 下载ControlNet 模型到 Google drive
    !wget -q -O cnmodels.py https://raw.githubusercontent.com/Van-wise/sd-colab/main/cnmodels.py
    from cnmodels import download
    from cnmodels import cndown_drive
    cndown_drive()
else:
    print("不使用 ControlNet、或已使用快捷方式。")

#链接 models
!test -d {webui_dir}/models/Stable-diffusion && test -d {gwebui_dir}/models/Stable-diffusion && ln -sf {gwebui_dir}/models/Stable-diffusion {webui_dir}/models/Stable-diffusion
model_dir = os.path.join(webui_dir, 'models', 'Stable-diffusion')
gmodel_dir = os.path.join(gwebui_dir, 'models', 'Stable-diffusion')
if not any(fname.endswith(('.safetensors', '.ckpt')) for fname in glob.glob(os.path.join(model_dir, '*'))) and not any(fname.endswith(('.safetensors', '.ckpt')) for fname in glob.glob(os.path.join(gmodel_dir, '*'))):
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors -d {model_dir} -o chilloutmix_NiPrunedFp32Fix.safetensors
else:
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/chilloutmix/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors -d {model_dir} -o chilloutmix_NiPrunedFp32Fix.safetensors
#链接 Lora
!test -d {webui_dir}/models/Lora && test -d {gwebui_dir}/models/Lora && ln -sf {gwebui_dir}/models/Lora {webui_dir}/models/Lora
Lora_dir = os.path.join(gwebui_dir, 'models', 'Lora')
gLora_dir = os.path.join(gwebui_dir, 'models', 'Lora')
if not any(fname.endswith(('.safetensors', '.ckpt')) for fname in glob.glob(os.path.join(Lora_dir, '*'))) and not any(fname.endswith(('.safetensors', '.ckpt')) for fname in glob.glob(os.path.join(gLora_dir, '*'))):
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/62833 -d {webui_dir}/models/Lora -o Detail_Tweaker_LoRA.safetensors ##Detail Tweaker LoRA/

#Stable-Diffusion
with capture.capture_output() as cap:
    %cd {gwebui_dir}/modules
    !wget -q -O extras.py https://raw.githubusercontent.com/AUTOMATIC1111/$sdw/master/modules/extras.py
    !wget -q -O sd_models.py https://raw.githubusercontent.com/AUTOMATIC1111/$sdw/master/modules/sd_models.py
    !wget -q -O /usr/local/lib/python3.10/dist-packages/gradio/blocks.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/blocks.py

    %cd {gwebui_dir}
    !sed -i 's@ui.create_ui().*@ui.create_ui();shared.demo.queue(concurrency_count=999999,status_update_rate=0.1)@' {gwebui_dir}/webui.py
    !sed -i 's@print(\"No module.*@@' /content/sd/stablediffusion/ldm/modules/diffusionmodules/model.py
    !sed -i 's@\["sd_model_checkpoint"\]@\["sd_model_checkpoint", "sd_vae", "CLIP_stop_at_last_layers", "inpainting_mask_weight", "initial_noise_multiplier"\]@g' {webui_dir}/modules/shared.py

    !sed -i 's@possible_sd_paths =.*@possible_sd_paths = [\"{gwebui_dir}/repositories/stable-diffusion-stability-ai\"]@' {gwebui_dir}/modules/paths.py
    !sed -i 's@\.\.\/@src/@g' {gwebui_dir}/modules/paths.py
    !sed -i 's@src/generative-models@generative-models@g' {gwebui_dir}/modules/paths.py

    !sed -i "s@os.path.splitext(checkpoint_file)@os.path.splitext(checkpoint_file); map_location='cuda'@"  {gwebui_dir}/modules/sd_models.py
    !sed -i "s@map_location='cpu'@map_location='cuda'@"  {gwebui_dir}/modules/extras.py
wise = "--share --api --disable-safe-unpickle --enable-insecure-extension-access --no-download-sd-model --no-half-vae --opt-sdp-attention --disable-console-progressbars --theme dark"

#@markdown ####❎是否使用[ngrok](https://dashboard.ngrok.com/get-started/your-authtoken)网络加速
#@markdown ######填写ngrok_auth 的 token：
ngrok_auth=""  #@param {type:"string"}
if ngrok_auth:
  wise+=f" --ngrok={ngrok_auth} --ngrok-region='auto'"
  wise+=f" --ngrok={ngrok_auth} --ngrok-region='auto'"

!python {gwebui_dir}/webui.py $wise --ckpt-dir {model_dir} --controlnet-dir {cn_dir}
