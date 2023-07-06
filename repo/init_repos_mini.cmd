@ECHO OFF

REM SAM
git clone https://github.com/facebookresearch/segment-anything

PUSHD segment-anything
IF NOT EXIST ckpt MKDIR ckpt
PUSHD ckpt

IF NOT EXIST sam_vit_b_01ec64.pth (wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

POPD
POPD

REM SegPGD
git clone https://github.com/u6630774/SegPGD

ECHO Done!
ECHO.

PAUSE
