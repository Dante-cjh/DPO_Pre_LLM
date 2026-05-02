CUDA_VISIBLE_DEVICES=5 nohup vllm serve Qwen/Qwen3-8B \
    --port 8000 \
    --max-model-len 2048 \
    --served-model-name Qwen3-8B \
    > logs/vllm.log 2>&1 &

echo "vLLM PID: $!"

# 等服务就绪（约 30-60s）
until curl -sf http://localhost:8000/health > /dev/null; do sleep 5; done
echo "vLLM ready"