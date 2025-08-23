#python3 run_mot_challenge.py \
###
 # @Author: hcxpami 503429928@qq.com
 # @Date: 2024-11-06 09:55:45
 # @LastEditors: hcxpami 503429928@qq.com
 # @LastEditTime: 2024-11-09 11:54:31
 # @FilePath: /TempRMOT_original/TrackEval/scripts/evaluate_rmot.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
#--METRICS HOTA \
#--SEQMAP_FILE /data/wudongming/MOTR/seqmap_rmot_clean.txt \
#--SKIP_SPLIT_FOL True \
#--GT_FOLDER /data/Dataset/MOT17/images/train \
#--TRACKERS_FOLDER /data/wudongming/MOTR/exps/rmot_v2a/results_epoch249_ \
#--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
#--TRACKERS_TO_EVAL /data/wudongming/MOTR/exps/rmot_v2a/results_epoch249_ \
#--USE_PARALLEL True \
#--NUM_PARALLEL_CORES 2 \
#--SKIP_SPLIT_FOL True \
#--PLOT_CURVES False
# export CUDA_VISIBLE_DEVICES='4,5,6'

python3 run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE /data2/lgy/RMOT/work3/TRMOT3_4/datasets/data_path/refer-kitti-v2/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER /data2/lgy/Dataset/RMOT/REFER-KITTI/Dataset/refer-kitti-v2/KITTI/training/image_02 \
--TRACKERS_FOLDER /data2/lgy/RMOT/work3/TRMOT3_4/exps/saved_models_rk/motion_1_v2/results_epoch99 \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL /data2/lgy/RMOT/work3/TRMOT3_4/exps/saved_models_rk/motion_1_v2/results_epoch99 \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

#python3 run_mot_challenge.py \
#--METRICS HOTA \
#--SEQMAP_FILE /data/wudongming/MOTR/seqmap_kitti_clean.txt \
#--SKIP_SPLIT_FOL True \
#--GT_FOLDER /data/Dataset/KITTI/training/image_02 \
#--TRACKERS_FOLDER /data/wudongming/FairMOT/exp/fairmot_kitti_2/result_epoch100 \
#--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
#--TRACKERS_TO_EVAL /data/wudongming/FairMOT/exp/fairmot_kitti_2/result_epoch100 \
#--USE_PARALLEL True \
#--NUM_PARALLEL_CORES 2 \
#--SKIP_SPLIT_FOL True \
#--PLOT_CURVES False