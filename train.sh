# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="lora"
MODEL="RETFound_mae"
# MODEL="Dinov3"
MODEL_ARCH="retfound_mae"
# MODEL_ARCH="dinov3_vith16plus"
FINETUNE="RETFound_mae_natureCFP"
# FINETUNE="dinov3_vith16plus_pretrain.pth"

# ==== Data settings ====
# change the dataset name and corresponding class number
#DATASET="MESSIDOR2"
DATASET="data/APTOS2019"
NUM_CLASS=5
data_path="./${DATASET}"
task="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --finetune "${FINETUNE}" \
  --savemodel \
  --global_pool \
  --batch_size 24 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${data_path}" \
  --input_size 224 \
  --task "${task}" \
  --adaptation "${ADAPTATION}" \
  --lora_rank 8 \
  --lora_alpha 16\
  --lora_dropout 0.0