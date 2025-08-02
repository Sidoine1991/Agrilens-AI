# AgriLens AI Makefile
# Common commands for development and deployment

.PHONY: help install setup dev test clean build run docker-build docker-run docker-stop logs monitor

# Default target
help:
	@echo "🌱 AgriLens AI - Available Commands"
	@echo "=================================="
	@echo "install    - Install dependencies"
	@echo "setup      - Run development setup script"
	@echo "dev        - Start development server"
	@echo "test       - Run tests"
	@echo "clean      - Clean cache and temporary files"
	@echo "build      - Build Docker image"
	@echo "run        - Run with Docker Compose"
	@echo "stop       - Stop Docker containers"
	@echo "logs       - View application logs"
	@echo "monitor    - Start performance monitoring"
	@echo "format     - Format code with black"
	@echo "lint       - Lint code with flake8"
	@echo "docs       - Generate documentation"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

# Run development setup
setup:
	@echo "⚙️  Running development setup..."
	python setup_dev.py

# Start development server
dev:
	@echo "🚀 Starting development server..."
	streamlit run src/streamlit_app_multilingual.py --server.port=8501

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest -v

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf logs/*.log
	rm -rf exports/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Build Docker image
build:
	@echo "🐳 Building Docker image..."
	docker build -t agrilens-ai .

# Run with Docker Compose
run:
	@echo "🚀 Starting AgriLens AI with Docker Compose..."
	docker-compose up -d

# Stop Docker containers
stop:
	@echo "🛑 Stopping Docker containers..."
	docker-compose down

# View application logs
logs:
	@echo "📋 Viewing application logs..."
	docker-compose logs -f agrilens-ai

# Start performance monitoring
monitor:
	@echo "📊 Starting performance monitoring..."
	python monitor_performance.py

# Format code
format:
	@echo "🎨 Formatting code with black..."
	black .

# Lint code
lint:
	@echo "🔍 Linting code with flake8..."
	flake8 .

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation is available in TECHNICAL_NOTE.md"

# Production deployment
deploy:
	@echo "🚀 Deploying to production..."
	docker-compose -f docker-compose.yml up -d --build

# Development environment
dev-env:
	@echo "🔧 Setting up development environment..."
	python setup_dev.py
	@echo "✅ Development environment ready!"

# Quick start (install + run)
quick-start:
	@echo "⚡ Quick start..."
	make install
	make dev

# Docker quick start
docker-quick:
	@echo "🐳 Docker quick start..."
	make build
	make run

# System check
check:
	@echo "🔍 Checking system requirements..."
	python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
	python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
	@echo "✅ System check completed"

# Backup models
backup:
	@echo "💾 Backing up models..."
	tar -czf models_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz models/

# Restore models
restore:
	@echo "📥 Restoring models..."
	@read -p "Enter backup file name: " backup_file; \
	tar -xzf $$backup_file

# Update dependencies
update:
	@echo "🔄 Updating dependencies..."
	pip install --upgrade -r requirements.txt

# Security check
security:
	@echo "🔒 Running security checks..."
	pip-audit
	safety check

# Performance test
perf-test:
	@echo "⚡ Running performance tests..."
	python monitor_performance.py --duration 300 --interval 10

# Help for Docker commands
docker-help:
	@echo "🐳 Docker Commands:"
	@echo "  make build        - Build image"
	@echo "  make run          - Start containers"
	@echo "  make stop         - Stop containers"
	@echo "  make logs         - View logs"
	@echo "  make clean-docker - Remove containers and images"

# Clean Docker
clean-docker:
	@echo "🧹 Cleaning Docker..."
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f 