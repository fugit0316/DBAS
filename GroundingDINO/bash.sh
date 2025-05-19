# CUDA_VISIBLE_DEVICES={0}

python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p weights/groundingdino_swint_ogc.pth \
-i vol/road_anomaly/original/*.jpg   \
-o "output/" \
-t "unknown object in the road"
 # [--cpu-only] # open it for cpu mode