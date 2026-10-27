bench_Ours-S:
	CUDA_VISIBLE_DEVICES=0  python model_benchmark.py --model Ours-S  --resume Ours-S
bench_Ours-L:
	CUDA_VISIBLE_DEVICES=1  python model_benchmark.py --model Ours-L  --resume Ours-L

# Usage: make demo-S I0=frame0.png I2=frame2.png [I1=gt_middle.png]
DEMO_ARGS = --i0 $(I0) --i2 $(I2) $(if $(I1),--i1 $(I1),)

demo-S:
	python demo.py --model Ours-S  --resume Ours-S $(DEMO_ARGS)
demo-L:
	python demo.py --model Ours-L  --resume Ours-L $(DEMO_ARGS)
