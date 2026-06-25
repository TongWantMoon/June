import os
import time
from typing import List, Dict, Any
'''所有 LLM 调用逻辑统一封装，其他模块只需调用 generate() 即可'''

class LLMInterface:

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, timeout: int = 30, **kwargs):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.openai.com/v1"))
        self.model = model or os.getenv("LLM_MODEL", "deepseek-v4-pro")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create OpenAI client: {exc}. "
                "Please install openai: pip install openai"
            ) from exc
        return self._client

    def generate(self, messages: List[Dict[str, str]], n: int = 1, stop=None, **kwargs) -> List[str]:

        client = self._ensure_client()
        stop = stop or []
        responses = []
        for _ in range(n):
            try:
                resp = client.chat.completions.create(
                    #发送请求给DeepSeek API
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=stop or None,
                )
                text = resp.choices[0].message.content.strip()
                responses.append(text)
            except Exception as exc:
                print(f"[LLMInterface] API call failed: {exc}")
                responses.append("")

            if len(responses) < n:
                time.sleep(0.5)
        return responses

    def __call__(self, prompts: List[str], **kwargs) -> List[str]:

        results = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            out = self.generate(messages, n=1, **kwargs)
            results.append(out[0] if out else "")
        return results


class DeepSeekInterface(LLMInterface):
    def __init__(self, api_key: str = None, model: str = "deepseek-v4-pro", **kwargs):
        super().__init__(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model=model,
            **kwargs,
        )


class MockLLMInterface:
    """无api时  测试用"""

    def __init__(self, **kwargs):
        self.api_key = "mock"
        self.base_url = "mock"
        self.model = "mock"
        self.temperature = 0.0
        self.max_tokens = 256
        self._step_counter = 0

    def generate(self, messages: List[Dict[str, str]], n: int = 1, stop=None, **kwargs) -> List[str]:

        """提取prompt内容"""
        prompt = ""
        for msg in messages:
            prompt += msg.get("content", "")

        self._step_counter += 1
        '''Critic模式'''
        if "critic" in prompt.lower() or "score" in prompt.lower():
            return ["3"]
        '''Action模式'''
        if "recommend" in prompt.lower() or "action" in prompt.lower():
            if "target" in prompt.lower():
                return [f"Thought {self._step_counter}: I should recommend the target item.\nAction {self._step_counter}: recommend[target_item]"]
            return [f"Thought {self._step_counter}: I should ask for preference.\nAction {self._step_counter}: ask[What genre do you prefer?]"]
        return [f"Thought {self._step_counter}: I need to analyze user intent.\nAction {self._step_counter}: clarify[Please specify your preference.]"]

    def __call__(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            out = self.generate(messages, n=1, **kwargs)
            results.append(out[0] if out else "")
        return results
