
bench_Ours-S:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-S  --resume Ours-S 
bench_Ours-L:
	CUDA_VISIBLE_DEVICES=1  python model_benchmark.py --model Ours-L  --resume Ours-L 

demo-S:
	python demo.py --model Ours-S  --resume Ours-S
demo-L:
	python demo.py --model Ours-L  --resume Ours-L
