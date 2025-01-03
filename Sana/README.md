```
pip install tabulate autoroot autorootcwd
```
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
  --nproc_per_node=2 --master_port=15432 \
  train_scripts/train_depth.py \
  --config_path=configs/sana_config/1024ms/Sana_600M_img1024.yaml \
  --work_dir=output/debug \
  --name=tmp \
  --resume_from=latest \
  --report_to=tensorboard \
  --debug=true \
  --model.load_from=hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth \
  --model.multi_scale=true \
  --train.train_batch_size=1
```
```bash
sh scripts/eval.sh \
  --base_data_dir=/data/om/data/eval_dataset \
  --dataset_config=configs/dataset/data_nyu_test.yaml \
  --output_dir=output/batch_eval/epoch_4_step_150000/nyu_test \
  --model_path=/data/om/Sana/output/debug/checkpoints/epoch_4_step_150000.pth \
  --config=configs/sana_config/1024ms/Sana_600M_img1024.yaml \
```