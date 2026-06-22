.PHONY: dev web train-curriculum train-curriculum-resume

# Khởi động FE + BE (reload tự động, mở trình duyệt)
dev:
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python dev.py; \
	else \
		python3 dev.py; \
	fi

# Curriculum DQN: Minimax depth 1→2→3→self-play
train-curriculum:
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python train_curriculum.py; \
	else \
		python3 train_curriculum.py; \
	fi

# Tiếp tục curriculum từ phase 2 (bỏ qua phase 1 đã xong, nạp dqn_15.pth)
train-curriculum-resume:
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python train_curriculum.py --start-phase 2; \
	else \
		python3 train_curriculum.py --start-phase 2; \
	fi

	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python web_main.py; \
	else \
		python3 web_main.py; \
	fi
