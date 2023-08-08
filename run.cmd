REM 无攻击，正常推理
REM 狗
python atk.py --point 400,200
REM 地板
python atk.py --point 400,400
REM 盆
python atk.py --point 250,550
REM 腿
python atk.py --point 200,700
REM 脚
python atk.py --point 400,600


REM 无目标攻击
REM e.g. 指针指向狗 → 预测出全黑
python atk.py --point 400,200
python atk.py --point 400,200 --step 40 --eps 0.1 --alpha 0.01

REM 无目标攻击 + 限制修改区域到 边缘
REM e.g. 指针指向狗 → 预测出全黑
python atk.py --point 400,200 --lim edge --thresh 0.1

REM 无目标攻击 + 限制修改区域到 目标 / 非目标 (目标的补)
REM e.g. 指针指向 狗, 仅修改 脚 区域
python atk.py --point 400,200 --lim 400,600
REM e.g. 指针指向 狗, 仅修改 狗 以外区域
python atk.py --point 400,200 --lim ~400,200


REM 有目标攻击
REM e.g. 指针指向狗 → 预测出盆子
python atk.py --point 400,200 --point_tgt 250,550

REM 有目标攻击 + 限制修改区域到 边缘
REM e.g. 指针指向 狗 → 预测出 盆子, 仅修改 边缘 区域
python atk.py --point 400,200 --point_tgt 250,550 --lim edge --thresh 0.1

REM 有目标攻击 + 限制修改区域到 目标
REM e.g. 指针指向 狗 → 预测出 盆子, 仅修改 狗 区域
python atk.py --point 400,200 --point_tgt 250,550 --lim 400,200
REM e.g. 指针指向 狗 → 预测出 盆子, 仅修改 脚 区域
python atk.py --point 400,200 --point_tgt 250,550 --lim 400,600


REM 跑整个 SAM 数据集
REM e.g. 干净预测
python atk_sam.py
REM e.g. 干净预测 (前 10 个样本)
python atk_sam.py -L 10
python atk_sam.py -L 10 --multi_mask
REM e.g. PGD 无目标攻击
python atk_sam.py --atk
REM e.g. PGD 无目标攻击，限制修改到 边缘
python atk_sam.py --atk --lim edge
REM e.g. PGD 有目标攻击 (随机目标)
python atk_sam.py --atk --tgt
REM e.g. PGD 有目标攻击 (随机目标) 限制修改到 该随机目标
python atk_sam.py --atk --tgt --lim tgt
REM e.g. PGD 有目标攻击 (随机目标) 限制修改到 非该随机目标
python atk_sam.py --atk --tgt --lim ~tgt


REM 跑整个 KITTI 数据集
REM e.g. 干净预测
python atk_kitti.py
REM e.g. 干净预测 (前 10 个样本)
python atk_kitti.py -L 10
python atk_kitti.py -L 10 --multi_mask
REM e.g. PGD 无目标攻击
python atk_kitti.py --atk
REM e.g. PGD 无目标攻击，限制修改到 边缘
python atk_kitti.py --atk --lim edge
REM e.g. PGD 有目标攻击 (随机目标)
python atk_kitti.py --atk --tgt
REM e.g. PGD 有目标攻击 (随机目标) 限制修改到 该随机目标
python atk_kitti.py --atk --tgt --lim tgt
REM e.g. PGD 有目标攻击 (随机目标) 限制修改到 非该随机目标
python atk_kitti.py --atk --tgt --lim ~tgt
