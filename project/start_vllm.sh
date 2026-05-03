CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.80 \
  --port 8000 \
  --served-model-name Qwen3-8B

echo "vLLM PID: $!"

# 等服务就绪（约 30-60s）
until curl -sf http://localhost:8000/health > /dev/null; do sleep 5; done
echo "vLLM ready"