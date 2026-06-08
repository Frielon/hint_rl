# inference env rebuild -> CUDA 12.8 (runs on node's 12.9 compat driver)
Reason: sglang 0.5.12.post1 stack (sgl_kernel/deep_gemm/torch2.11) was CUDA-13-built;
node driver is 550.x -> CUDA 12.9 max. Downgraded to newest sglang with cu128 kernels.

Target (python 3.11):
  torch==2.8.0+cu128  torchvision==0.23.0+cu128  torchaudio==2.8.0+cu128   (download.pytorch.org/whl/cu128)
  sgl-kernel==0.3.14.post1+cu128                                            (docs.sglang.ai/whl/cu128)
  sglang[all]==0.5.3                                                        (PyPI; pulls flashinfer==0.4.0rc3)
Backups: inference_freeze_cu130.txt, inference_freeze_cu128.txt
