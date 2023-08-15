REM 攻击示例图中几个关键物体
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


REM 对比 干净样本 和 对抗样本 的 CAM 图
REM 无目标:狗
python atk_vis.py --point 400,200 --cam_meth GradCAM
REM 有目标: 狗 → 盆子
python atk_vis.py --point 400,200 --point_tgt 250,550 --cam_meth GradCAM


REM 通用攻击配置参数
REM 1. 攻击方法
python atk.py --point 400,200 --meth SegPGD
REM 2. 攻击强度
python atk.py --point 400,200 --step 40 --eps 0.1 --alpha 0.01


REM 无目标攻击：e.g. 指针指向 狗 → 预测出 全黑
REM 命令行比有目标攻击多 --loss_w, --force_bce 两个 **互斥的** 可选选项
python atk.py --point 400,200
python atk.py --point 400,200 --loss_w -4
python atk.py --point 400,200 --force_bce

REM 无目标攻击 + 限制修改区域
REM 1. 仅修改 边缘
python atk.py --point 400,200 --lim edge --edge_w 0.1
REM 2. 仅修改 Saliency Map 区域
python atk.py --point 400,200 --lim smap --smap_w 0.5
REM 3. 仅修改 CAM 区域
python atk.py --point 400,200 --lim cam --cam_w 0.1 --cam_meth GradCAM
REM 4. 仅修改 lim点原始预测 区域 (脚)
python atk.py --point 400,200 --lim 400,600
REM 5. 仅修改 lim点原始预测 以外的区域 (非狗)
python atk.py --point 400,200 --lim ~400,200


REM 有目标攻击：e.g. 指针指向 狗 → 预测出 盆子
REM 命令行比无目标攻击多一个 --point_tgt 必选选项
python atk.py --point 400,200 --point_tgt 250,550

REM 有目标攻击 + 限制修改区域
REM 1. 仅修改 边缘
python atk.py --point 400,200 --point_tgt 250,550 --lim edge --edge_w 0.1
REM 2. 仅修改 Saliency Map 区域
python atk.py --point 400,200 --point_tgt 250,550 --lim smap --smap_w 0.5
REM 3. 仅修改 CAM 区域
python atk.py --point 400,200 --point_tgt 250,550 --lim cam --cam_w 0.1 --cam_meth GradCAM
REM 4. 仅修改 lim点原始预测 区域 (脚)
python atk.py --point 400,200 --point_tgt 250,550 --lim 400,600
REM 5. 仅修改 lim点原始预测 以外的区域 (非狗)
python atk.py --point 400,200 --point_tgt 250,550 --lim ~400,200



REM 跑整个 SAM 数据集
REM e.g. 干净预测
python atk_sam.py
python atk_sam.py -L 10
python atk_sam.py -L 10 --multi_mask
REM e.g. 无目标攻击
python atk_sam.py --atk
REM e.g. 无目标攻击，限制修改到 边缘
python atk_sam.py --atk --lim edge
REM e.g. 有目标攻击 (随机目标)
python atk_sam.py --atk --tgt
REM e.g. 有目标攻击 (随机目标) 限制修改到 该随机目标
python atk_sam.py --atk --tgt --lim tgt
REM e.g. 有目标攻击 (随机目标) 限制修改到 非该随机目标
python atk_sam.py --atk --tgt --lim ~tgt


REM 跑整个 KITTI 数据集
REM e.g. 干净预测
python atk_kitti.py
python atk_kitti.py -L 10
python atk_kitti.py -L 10 --multi_mask
REM e.g. 无目标攻击
python atk_kitti.py --atk
REM e.g. 无目标攻击，限制修改到 边缘
python atk_kitti.py --atk --lim edge
REM e.g. 有目标攻击 (随机目标)
python atk_kitti.py --atk --tgt
REM e.g. 有目标攻击 (随机目标) 限制修改到 该随机目标
python atk_kitti.py --atk --tgt --lim tgt
REM e.g. 有目标攻击 (随机目标) 限制修改到 非该随机目标
python atk_kitti.py --atk --tgt --lim ~tgt
