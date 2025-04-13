
bench_Ours-S:
	CUDA_VISIBLE_DEVICES=1  python model_benchmark.py --model Ours-S  --resume Ours-S --bench [UCF101]
bench_Ours-L:
	CUDA_VISIBLE_DEVICES=1  python model_benchmark.py --model Ours-L  --resume Ours-L --bench [UCF101]

demo-S:
	python demo.py --model Ours-S  --resume Ours-S
demo-L:
	python demo.py --model Ours-L  --resume Ours-L
