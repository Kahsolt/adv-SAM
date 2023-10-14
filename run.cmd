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
python vis_atk_cam.py --point 400,200 --cam_meth GradCAM
REM 有目标: 狗 → 盆子
python vis_atk_cam.py --point 400,200 --point_tgt 250,550 --cam_meth GradCAM


REM 通用攻击配置参数
REM 1. 攻击方法
python atk.py --point 400,200 --meth SegPGD
REM 2. 损失函数
python atk.py --point 400,200 --loss BCE
REM 3. 攻击强度
python atk.py --point 400,200 --steps 40 --eps 0.1 --alpha 0.01


REM 无目标攻击：e.g. 指针指向 狗 → 预测出 全黑
REM 命令行比有目标攻击多一个 --loss_w 选项
python atk.py --point 400,200
python atk.py --point 400,200 --loss_w -4
REM 只有 BCE 不需要 --loss_w, 其他都需要
python atk.py --point 400,200 --loss BCE

REM 有目标攻击：e.g. 指针指向 狗 → 预测出 盆子
REM 命令行比无目标攻击多一个 --point_tgt 必选选项
python atk.py --point 400,200 --point_tgt 250,550

REM 跨图像攻击
python atk.py --point 400,200 --point_tgt 250,550 --f_tgt repo\segment-anything\notebooks\images\groceries.jpg


REM 跑整个 SAM 数据集
REM e.g. 干净预测
python atk_sam.py
python atk_sam.py -L 10
python atk_sam.py -L 10 --multi_mask
REM e.g. 无目标攻击
python atk_sam.py --atk
REM e.g. 有目标攻击 (随机目标)
python atk_sam.py --atk --tgt


REM 跑整个 KITTI 数据集
REM e.g. 干净预测
python atk_kitti.py
python atk_kitti.py -L 10
python atk_kitti.py -L 10 --multi_mask
REM e.g. 无目标攻击
python atk_kitti.py --atk
REM e.g. 有目标攻击 (随机目标)
python atk_kitti.py --atk --tgt
