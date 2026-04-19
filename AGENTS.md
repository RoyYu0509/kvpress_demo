# AGENTS.md

This project must run on an Apple Silicon MacBook Pro.
Use MPS if available, otherwise CPU.
Do not assume CUDA.
Do not use flash-attn, bitsandbytes, vllm, xformers, or triton.
Keep everything small and practical for a same-day demo.
Use seq_len <= 256 and batch size 1 unless necessary.
Save outputs under ./outputs.
For complex tasks, propose a plan first before implementing.
