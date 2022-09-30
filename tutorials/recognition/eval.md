# 模型验证

MindFace支持人脸识别模型在多个验证集上验证模型性能，当前支持的人脸验证数据集包括lfw,cfp_fp,agedb_30,calfw,cplfw

## 验证数据集

MindFace人脸验证数据集使用.bin格式，所有的验证集全部放在同一目录下，数据集的组织结构如下所示：

```
val_dataset
 ├── agedb_30.bin
 ├── calfw.bin
 ├── cfp_fp.bin
 ├── cplfw.bin
 └── lfw.bin
```

### 执行验证程序

GPU平台

```
sh scripts/run_eval_gpu.sh /path/to/dataset /path/to/ckpt model_name
```

Ascend平台

```
sh scripts/run_eval.sh /path/to/dataset /path/to/ckpt model_name
```