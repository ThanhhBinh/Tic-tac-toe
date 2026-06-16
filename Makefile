.PHONY: dev web

# Khởi động FE + BE (reload tự động, mở trình duyệt)
dev:
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python dev.py; \
	else \
		python3 dev.py; \
	fi

# Chạy production-like (không reload)
web:
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python web_main.py; \
	else \
		python3 web_main.py; \
	fi
