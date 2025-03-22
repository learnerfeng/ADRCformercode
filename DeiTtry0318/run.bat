@echo off
:: 设置 PyTorch 显存分配环境变量，减少显存碎片化
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

:: BASELINE
python E:/ProjectTestDeeplearing/DeiT/main.py  ^
--model deit_tiny_patch16_224 ^
--batch-size 32 ^
--data-path E:/ProjectTestDeeplearing/DeiT/datasetdeit ^
--output_dir E:/ProjectTestDeeplearing/DeiT/output_baseline

:: NEUTRENO
python E:/ProjectTestDeeplearing/DeiT/main.py  ^
--model deit_tiny_neutreno_patch16_224 ^
--batch-size 32 ^
--data-path E:/ProjectTestDeeplearing/DeiT/datasetdeit ^
--output_dir E:/ProjectTestDeeplearing/DeiT/output_neutreno
