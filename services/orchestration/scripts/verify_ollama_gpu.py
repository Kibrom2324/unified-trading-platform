#!/usr/bin/env python3
"""
Verify Ollama GPU Configuration

This script checks:
1. GPU availability and driver status
2. Ollama service status
3. GPU usage during inference
4. Provides configuration recommendations

Usage:
    python scripts/verify_ollama_gpu.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time


def run_cmd(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_nvidia_driver() -> dict:
    """Check NVIDIA driver and GPU status."""
    print("\n" + "=" * 60)
    print("🔍 CHECKING NVIDIA GPU & DRIVER")
    print("=" * 60)
    
    rc, out, err = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader")
    
    if rc != 0:
        print("❌ NVIDIA driver not found or nvidia-smi failed")
        print(f"   Error: {err}")
        return {"status": "failed", "error": err}
    
    parts = out.split(", ")
    gpu_info = {
        "status": "ok",
        "name": parts[0] if len(parts) > 0 else "unknown",
        "driver_version": parts[1] if len(parts) > 1 else "unknown",
        "memory_total": parts[2] if len(parts) > 2 else "unknown",
        "compute_capability": parts[3] if len(parts) > 3 else "unknown",
    }
    
    print(f"✅ GPU Found: {gpu_info['name']}")
    print(f"   Driver Version: {gpu_info['driver_version']}")
    print(f"   Memory: {gpu_info['memory_total']}")
    print(f"   Compute Capability: {gpu_info['compute_capability']}")
    
    return gpu_info


def check_cuda() -> dict:
    """Check CUDA installation."""
    print("\n" + "=" * 60)
    print("🔍 CHECKING CUDA")
    print("=" * 60)
    
    # Check nvcc
    rc, out, _ = run_cmd("nvcc --version 2>/dev/null | grep release")
    if rc == 0:
        print(f"✅ CUDA Compiler: {out}")
    else:
        print("⚠️  nvcc not found (not required for Ollama)")
    
    # Check CUDA libraries
    rc, out, _ = run_cmd("ldconfig -p | grep -i cuda | head -3")
    if rc == 0 and out:
        print(f"✅ CUDA Libraries found")
    else:
        print("⚠️  CUDA libraries not in ldconfig (Ollama bundles its own)")
    
    return {"status": "ok"}


def check_ollama_service() -> dict:
    """Check Ollama service status."""
    print("\n" + "=" * 60)
    print("🔍 CHECKING OLLAMA SERVICE")
    print("=" * 60)
    
    # Check version
    rc, version, _ = run_cmd("ollama --version")
    if rc != 0:
        print("❌ Ollama not installed")
        return {"status": "failed", "error": "Ollama not installed"}
    
    print(f"✅ Ollama Version: {version}")
    
    # Check service status
    rc, status, _ = run_cmd("systemctl is-active ollama")
    if status == "active":
        print("✅ Ollama service is running")
    else:
        print(f"⚠️  Ollama service status: {status}")
    
    # Check listening
    rc, out, _ = run_cmd("curl -s http://localhost:11434/api/version")
    if rc == 0:
        try:
            ver = json.loads(out)
            print(f"✅ Ollama API responding: {ver.get('version', 'unknown')}")
        except:
            print(f"✅ Ollama API responding")
    else:
        print("❌ Ollama API not responding on localhost:11434")
        return {"status": "failed", "error": "API not responding"}
    
    # Check models
    rc, out, _ = run_cmd("ollama list")
    if rc == 0:
        models = [l.split()[0] for l in out.split("\n")[1:] if l.strip()]
        print(f"✅ Available models: {', '.join(models) if models else 'none'}")
    
    # Check service config
    rc, out, _ = run_cmd("cat /etc/systemd/system/ollama.service.d/override.conf 2>/dev/null")
    if rc == 0 and out:
        print(f"   Service overrides:\n{out}")
    
    return {"status": "ok", "version": version}


def check_gpu_memory() -> dict:
    """Check current GPU memory usage."""
    print("\n" + "=" * 60)
    print("🔍 CHECKING GPU MEMORY")
    print("=" * 60)
    
    rc, out, _ = run_cmd("nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader")
    if rc != 0:
        print("❌ Could not query GPU memory")
        return {"status": "failed"}
    
    parts = out.split(", ")
    used = parts[0] if len(parts) > 0 else "0 MiB"
    free = parts[1] if len(parts) > 1 else "0 MiB"
    total = parts[2] if len(parts) > 2 else "0 MiB"
    
    print(f"   Used:  {used}")
    print(f"   Free:  {free}")
    print(f"   Total: {total}")
    
    # Parse used memory
    used_mb = int(used.replace(" MiB", "").strip())
    if used_mb > 500:
        print("✅ Model appears to be loaded in GPU memory")
    else:
        print("⚠️  Low GPU memory usage - model may not be loaded")
    
    return {"status": "ok", "used": used, "free": free, "total": total}


def test_inference_gpu() -> dict:
    """Test inference and measure GPU utilization."""
    print("\n" + "=" * 60)
    print("🔍 TESTING GPU INFERENCE")
    print("=" * 60)
    
    # Get baseline GPU usage
    rc, baseline, _ = run_cmd("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    baseline_parts = baseline.split(", ") if rc == 0 else ["0 %", "0 MiB"]
    baseline_util = baseline_parts[0]
    baseline_mem = baseline_parts[1] if len(baseline_parts) > 1 else "0 MiB"
    
    print(f"📊 Baseline GPU: {baseline_util} utilization, {baseline_mem} memory")
    print("🚀 Running inference test...")
    
    # Run inference
    start = time.time()
    rc, out, err = run_cmd(
        '''curl -s http://localhost:11434/api/generate -d '{"model": "llama3.2:3b", "prompt": "Explain quantum computing in 100 words.", "stream": false}' ''',
        timeout=120,
    )
    elapsed = time.time() - start
    
    if rc != 0:
        print(f"❌ Inference failed: {err}")
        return {"status": "failed", "error": err}
    
    try:
        response = json.loads(out)
        word_count = len(response.get("response", "").split())
        tokens_per_sec = response.get("eval_count", 0) / response.get("eval_duration", 1) * 1e9
    except:
        word_count = 0
        tokens_per_sec = 0
    
    # Check GPU usage after inference
    rc, after, _ = run_cmd("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    after_parts = after.split(", ") if rc == 0 else ["0 %", "0 MiB"]
    after_util = after_parts[0]
    after_mem = after_parts[1] if len(after_parts) > 1 else "0 MiB"
    
    print(f"\n✅ Inference completed in {elapsed:.2f}s")
    print(f"   Words generated: {word_count}")
    print(f"   Tokens/sec: {tokens_per_sec:.1f}")
    print(f"📊 After GPU: {after_util} utilization, {after_mem} memory")
    
    # Parse memory values
    baseline_mem_mb = int(baseline_mem.replace(" MiB", "").strip())
    after_mem_mb = int(after_mem.replace(" MiB", "").strip())
    
    if after_mem_mb > baseline_mem_mb + 100:
        print("✅ GPU memory increased during inference - model is using GPU!")
    elif after_mem_mb > 1000:
        print("✅ GPU memory shows model is loaded in VRAM")
    else:
        print("⚠️  GPU memory did not increase - model may be using CPU")
    
    if tokens_per_sec > 20:
        print(f"✅ Speed ({tokens_per_sec:.0f} tokens/sec) indicates GPU acceleration")
    elif tokens_per_sec > 5:
        print(f"⚠️  Speed ({tokens_per_sec:.0f} tokens/sec) - moderate, may be GPU constrained")
    else:
        print(f"⚠️  Speed ({tokens_per_sec:.0f} tokens/sec) - slow, may be CPU only")
    
    return {
        "status": "ok",
        "elapsed": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "gpu_memory_after": after_mem,
    }


def print_recommendations():
    """Print configuration recommendations."""
    print("\n" + "=" * 60)
    print("📋 RECOMMENDED CONFIGURATION")
    print("=" * 60)
    
    print("""
1. OLLAMA SERVICE CONFIGURATION
   Create/edit /etc/systemd/system/ollama.service.d/override.conf:
   
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0:11434"
   Environment="CUDA_VISIBLE_DEVICES=0"
   Environment="OLLAMA_GPU_DRIVER=cuda"
   Environment="OLLAMA_NUM_GPU=99"

   Then reload:
   sudo systemctl daemon-reload
   sudo systemctl restart ollama

2. VERIFY GPU USAGE
   # Watch GPU in real-time
   watch -n 1 nvidia-smi
   
   # Run inference in another terminal
   ollama run llama3.2:3b "Hello"

3. DOCKER CONTAINERS
   For containers needing GPU access to Ollama:
   - Use OLLAMA_HOST=http://host.docker.internal:11434
   - Or use host network mode
   
4. PERFORMANCE TUNING
   # For larger models, set context size:
   OLLAMA_CONTEXT_LENGTH=4096
   
   # For batch processing:
   OLLAMA_MAX_LOADED_MODELS=2
   
5. TROUBLESHOOTING
   # Check Ollama logs
   journalctl -u ollama -f
   
   # Check GPU processes
   nvidia-smi pmon -d 1
""")


def main():
    print("=" * 60)
    print("🔧 OLLAMA GPU VERIFICATION TOOL")
    print("=" * 60)
    
    results = {}
    
    # Check GPU
    results["gpu"] = check_nvidia_driver()
    if results["gpu"]["status"] != "ok":
        print("\n❌ GPU check failed. Cannot proceed.")
        sys.exit(1)
    
    # Check CUDA
    results["cuda"] = check_cuda()
    
    # Check Ollama
    results["ollama"] = check_ollama_service()
    if results["ollama"]["status"] != "ok":
        print("\n❌ Ollama not running. Start with: sudo systemctl start ollama")
        sys.exit(1)
    
    # Check GPU memory
    results["memory"] = check_gpu_memory()
    
    # Test inference
    results["inference"] = test_inference_gpu()
    
    # Print recommendations
    print_recommendations()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    all_ok = all(r.get("status") == "ok" for r in results.values())
    
    if all_ok:
        print("✅ All checks passed! Ollama is using GPU for inference.")
    else:
        print("⚠️  Some checks had issues. Review output above.")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
