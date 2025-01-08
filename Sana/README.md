```bash
python -m torch.distributed.run \
  --nproc_per_node=2 --master_port=15432 \
  train_scripts/train_depth.py \
  --config_path=configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
  --work_dir=output/debug \
  --name=tmp \
  --resume_from=latest \
  --report_to=tensorboard \
  --debug=true \
  --data.data_dir="[asset/example_data]" \
  --data.type=SanaImgDataset \
  --model.load_from=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth \
  --model.multi_scale=false \
  --train.train_batch_size=2
```
```bash
python -m torch.distributed.run \
  --nproc_per_node=1 --master_port=15432 \
  train_scripts/train_depth.py \
  --config_path=configs/sana_config/1024ms/Sana_1600M_img1024_AdamW.yaml \
  --work_dir=output/full_train_1600_1024 \
  --name=tmp \
  --resume_from=latest \
  --report_to=tensorboard \
  --debug=true \
  --model.load_from=hf://Efficient-Large-Model/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth \
  --model.multi_scale=true \
  --train.train_batch_size=1 \
  --base_data_dir=somewhere/else
```
