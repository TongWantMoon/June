"""DeepSeek API连通性测试，带超时"""
# B阶段API连通测试，检查网络、DNS、openai包版本，发一个最小请求验证API可用
import sys, os, socket

TPRA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'T-PRA-main'))
WORK_ROOT = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for r in (TPRA_ROOT, WORK_ROOT):
    if r not in sys.path:
        sys.path.insert(0, r)

import argparse


def test_connectivity(api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat", timeout: int = 30):
    print("=" * 60)
    print(f"DeepSeek API Quick Test")
    print(f"  base_url: {base_url}")
    print(f"  model: {model}")
    print(f"  timeout: {timeout}s")
    print("=" * 60)

    # 1. 检查openai包是否安装
    try:
        import openai
        print(f"[OK] openai package version: {openai.__version__}")
    except ImportError:
        print("[FAIL] openai package not installed. Run: pip install openai")
        return False

    # 2. 测试DNS解析和连通性
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(base_url)
        host = parsed.hostname or "api.deepseek.com"
        port = parsed.port or 443
        print(f"[INFO] Resolving {host}...")
        addr = socket.getaddrinfo(host, port)[0][4][0]
        print(f"[OK] DNS resolved: {host} -> {addr}")
    except Exception as e:
        print(f"[WARN] DNS resolution failed: {e}")

 # 3. 创建客户端，发最小请求
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        print(f"[INFO] Sending a minimal chat request...")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1024,
            temperature=0.0,
            # 如果需要用V4的thinking模式，通过extra_body传入（可选）
            #extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
        )
        print(f"[OK] API responded successfully.")
        print(f"[INFO] Response content: {resp.choices[0].message.content!r}")
        print(f"[INFO] Usage: prompt_tokens={resp.usage.prompt_tokens}, completion_tokens={resp.usage.completion_tokens}")
        return True
    except Exception as e:
        print(f"[FAIL] API call failed: {e}")
        return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] No API key provided. Set DEEPSEEK_API_KEY or pass --api_key.")
        sys.exit(1)

    success = test_connectivity(args.api_key, model=args.model, timeout=args.timeout)
    sys.exit(0 if success else 1)
