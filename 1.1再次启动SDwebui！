#@title 👇 1.1再次启动SDwebui！
#此单元格的用法是，在运行过1. 的前提下（安装好环境），启动SDweibui。
import os
import sys

import binascii
sdw = binascii.unhexlify("737461626c652d646966667573696f6e2d7765627569").decode('ascii')
w = binascii.unhexlify("73642d7765627569").decode('ascii')
webui_dir = f'/content/{sdw}'
gwebui_dir = f'/content/drive/MyDrive/{sdw}'
cn_dir = f"{webui_dir}/extensions/{w}-controlnet/models"
gcn_dir = f"{gwebui_dir}/extensions/{w}-controlnet/models"

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

    !sed -i 's@possible_sd_paths =.*@possible_sd_paths = [\"/content/sd/stablediffusion/\"]@' {gwebui_dir}/modules/paths.py
    !sed -i 's@\.\.\/@src/@g' {gwebui_dir}/modules/paths.py
    !sed -i 's@src/generative-models@generative-models@g' {gwebui_dir}/modules/paths.py

wise = "--share --api --disable-safe-unpickle --enable-insecure-extension-access --no-download-sd-model --no-half-vae --opt-sdp-attention --disable-console-progressbars --theme dark"
!python {gwebui_dir}/webui.py $wise --ckpt-dir {model_dir} --controlnet-dir {cn_dir}
