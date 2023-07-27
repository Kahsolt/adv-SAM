REM 狗
python atk_sam.py --point 400,200

REM 地板
python atk_sam.py --point 400,400

REM 盆
python atk_sam.py --point 250,550

REM 腿
python atk_sam.py --point 200,700

REM 脚
python atk_sam.py --point 400,600



REM 限制攻击区域 到 边缘算子检出区
python atk_sam.py --point 400,200 --lim --thresh 0.1 --step 40 --eps 0.1 --alpha 0.01


REM 有目标攻击： 指针指向狗 → 预测出盆子
python atk_sam.py --point 400,200 --point_tgt 250,550


REM 限制到边缘 + 有目标攻击
python atk_sam.py --point 400,200 --point_tgt 250,550 --lim
