@ECHO OFF

PUSHD %~dp0

REM SAM
git clone https://github.com/facebookresearch/segment-anything

PUSHD segment-anything
IF NOT EXIST ckpt MKDIR ckpt
PUSHD ckpt

IF NOT EXIST sam_vit_b_01ec64.pth (wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

IF NOT "%1"=="all" GOTO skip_large_models
IF NOT EXIST sam_vit_l_0b3195.pth (wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
IF NOT EXIST sam_vit_h_4b8939.pth (wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
:skip_large_models

POPD
POPD

REM HQ-SAM (Jun 1, 2023)
git clone https://github.com/SysCV/sam-hq

REM FastSAM (Jun 21, 2023)
git clone https://github.com/CASIA-IVA-Lab/FastSAM

REM MobileSAM (Jun 26, 2023)
git clone https://github.com/ChaoningZhang/MobileSAM

REM EfficientSAM (Nov 28, 2023)
git clone https://github.com/yformer/EfficientSAM

REM TinySAM (Dec 19, 2023)
git clone https://github.com/xinghaochen/TinySAM


REM Grad-CAM
git clone https://github.com/jacobgil/pytorch-grad-cam

REM SegPGD
git clone https://github.com/u6630774/SegPGD

REM robust-segmentation
git clone https://github.com/nmndeep/robust-segmentation

POPD

ECHO Done!
ECHO.

PAUSE
