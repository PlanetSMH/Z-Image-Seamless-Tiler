# DiT-Seamless-Tiler

Seamless tiling toolkit for DiT-based image generation models: Z-image, Qwen-Image, Flux

This repository focuses on practical methods for reducing seam artifacts in tileable textures:

- Latent/Noise rolling during denoising
- Circular VAE decoder padding

### Current Status

- **txt2img(Z-Image)**: Available
- **txt2img(Flux2Klein_4B)**: Available
- **txt2img(Qwen-Image)**: **Coming Soon**

- **img2img**: **Coming Soon**

### Environments

- **Z-Image** scripts run in [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).
- **Flux** scripts run in a local environment with `diffusers + torch`.

### Repository Layout

```text
.
├── assets/
│   ├── txt2img/
│   │   ├── Qwen-Image/
│   │   ├── Z-Image/
│   │   └── Flux2Klein_4B/
│   └── img2img/
├── txt2img/
│   ├── Qwen-Image/
│   ├── Z-Image/
│   │   └── z_image_tiling.py
│   └── Flux2Klein_4B/
│       └── flux_txt2img_seamless.py
└── img2img/
```

### Usage (Flux txt2img)

```bash
python txt2img/Flux2Klein_4B/flux_txt2img_seamless.py --model-dir /path/to/Model
```

### Visual Results

#### Z-Image txt2img

| Original (2x2) | Seamless (2x2) |
| :---: | :---: |
| ![Z-Image Original](assets/txt2img/Z-Image/Z-Image-Original.png) | ![Z-Image Seamless](assets/txt2img/Z-Image/Z-Image-Seamless-Tiler.png) |

#### Flux2Klein_4B txt2img

| Original (2x2) | Seamless (2x2) |
| :---: | :---: |
| ![Flux Original](assets/txt2img/Flux2Klein_4B/Flux2Klein_4B_Original.png) | ![Flux Seamless](assets/txt2img/Flux2Klein_4B/Flux2Klein_4B_Seamless_Tiler.png) |

---


## Acknowledgements

This project was initiated during my internship at Ubisoft China La Forge.
I sincerely appreciate the team's support, guidance, and the research environment they provided.

The open-source code and documentation in this repository are independently organized and maintained by me.
You can look forward to more mature official open-source implementations and ComfyUI workflow solutions from Ubisoft China La Forge in the future.


------------------------------------------------------------------------------------------------------------


### 当前进度


- **txt2img(Z-Image)**: 已支持
- **txt2img(Flux2Klein_4B)**: 已支持
- **txt2img(Qwen-Image)**: **即将上线**

- **img2img**: **即将上线**

### 环境要求

- **Z-Image** 脚本运行在 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 环境。
- **Flux** 脚本运行在本地 `diffusers + torch` 环境。

### Z-Image 使用方式（保持原有方式）

请将 `txt2img/Z-Image/z_image_tiling.py` 放到 DiffSynth-Studio 项目的对应推理目录中运行（与原项目使用方式一致）。

## 致谢

本项目起始于我在育碧中国 La Forge 实习期间的相关任务探索。
感谢团队提供的研究环境、指导与支持。

本仓库中的开源代码与文档由我独立整理与维护。
后续可期待育碧中国 La Forge 官方发布更成熟的开源项目实现及 ComfyUI 工作流方案。
