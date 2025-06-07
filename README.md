# 环境配置

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install blobfile piq matplotlib opencv-python joblib lmdb scipy clean-fid easydict torchmetrics rich ipdb wandb
```

# 数据获取
执行以下脚本下载数据集
```
cd assets/datasets
bash download_extract_edges2handbags.sh
```

## 模型权重
使用  [DDBM](https://github.com/alexzhou907/DDBM) 的预训练模型，请将模型权重置于  `assets/ckpts/` 目录。

下载：[e2h_ema_0.9999_420000.pt](https://huggingface.co/alexzhou907/DDBM/resolve/main/e2h_ema_0.9999_420000.pt)

此后执行：`python preprocess_ckpt.py` 

## Evaluations

下载参考数据，置于 `assets/stats/`:

[edges2handbags_ref_64_data.npz](https://huggingface.co/alexzhou907/DDBM/resolve/main/edges2handbags_ref_64_data.npz).

此后运行评测脚本

```
bash scripts/evaluate.sh $NFE $SAMPLER
```

其中 `$SAMPLER` 取值：
- ddbm
- dpmsolver1
- dpmsolver2
- dpmsolver3
