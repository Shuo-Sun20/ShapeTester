"""LLM client wrapper with caching and prompt utilities."""

import hashlib
import json
import os
from pathlib import Path
from re import sub
from typing import Any, Dict, Optional

from openai import OpenAI, OpenAIError

from utils.config import API_KEY, BASE_URL, MODEL_NAME, llm_cache_path
from utils.llm_client.prompt_config import PromptConfig
from utils.log_manager import setup_logger
from utils import testcase_to_str

class LLMClient:
    """Simple DeepSeek chat client with prompt helpers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        # Resolve configuration from environment variables or defaults
        self.api_key = api_key or os.getenv("LLM_API_KEY", API_KEY)
        self.model = model or os.getenv("LLM_MODEL_NAME", MODEL_NAME)
        self.base_url = base_url or os.getenv("LLM_BASE_URL", BASE_URL)
        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        self.logger = setup_logger(sub_dir="llm")
        self.logger.info(f"LLMClient initialized: model={self.model}, base_url={self.base_url}")
        self.last_reason: Optional[str] = None
        # Ensure cache directory exists
        Path(llm_cache_path).mkdir(parents=True, exist_ok=True)

    
    def chat(
        self,
        prompt: str,
        sub_dir: str = "default",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        pay_load: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat request and return content; use cache when possible."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        cache_key = self._build_cache_key(
            pay_load=pay_load,
        )
        cached = self._read_cache(cache_key, sub_dir=sub_dir)
        if cached is not None:
            self.last_reason = cached.get("reason")
            self.logger.info("Cache hit for LLM request; returning cached content")
            return cached.get("content", "")

        self.logger.debug("Sending chat completion request")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except OpenAIError as exc:  # noqa: BLE001
            self.logger.error(f"LLM request failed: {exc}")
            raise RuntimeError("LLM request failed") from exc

        if not resp.choices:
            self.logger.error("LLM response missing choices")
            raise RuntimeError("LLM response missing choices")
        # extract optional reasoning
        reason: Optional[str] = None
        try:
            # attempt to read custom fields directly
            reason = getattr(resp.choices[0].message, "reason", None) or getattr(
                resp.choices[0].message, "reasoning", None
            )
        except Exception:
            reason = None
        # fallback: dump to dict and probe common keys
        if reason is None:
            try:
                data = resp.model_dump()
                msg = (
                    data.get("choices", [{}])[0].get("message", {})
                    if isinstance(data, dict)
                    else {}
                )
                reason = msg.get("reason") or msg.get("reasoning") or msg.get("reasoning_content")
            except Exception:
                reason = None
        self.last_reason = reason
        self.logger.info(f"LLM reasoning: {reason if reason is not None else 'None'}")

        content = resp.choices[0].message.content
        if content is None:
            self.logger.error("LLM response missing content")
            raise RuntimeError("LLM response missing content")
        self.logger.debug(f"LLM content received (truncated): {str(content)[:200]}")
        self._write_cache(cache_key, {"pay_load": pay_load, "prompt": prompt, "content": content, "reason": self.last_reason}, sub_dir=sub_dir)
        return content

    def run_prompt_config(
        self,
        prompt_cfg: PromptConfig,
        sub_dir: str = "default",
        use_cn: bool = False,
        format_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a PromptConfig with retries, extraction and validation."""
        self.use_cn = use_cn
        fmt = prompt_cfg.prompt_CN if use_cn else prompt_cfg.prompt_EN
        pay_load = format_kwargs or {}
        prompt = fmt.format(**(format_kwargs or {}))
        self.logger.info(
            f"Running PromptConfig (CN={use_cn}) with retries={prompt_cfg.retry_times}"
        )

        result: Optional[Any] = None
        last_error: Optional[Exception] = None
        attempts = max(1, prompt_cfg.retry_times)
        current_prompt = prompt
        errorMsg_list = []
        for attempt in range(1, attempts + 1):
            try:
                self.logger.debug(f"Prompt attempt {attempt}/{attempts}")
                raw = self.chat(current_prompt, sub_dir=sub_dir+f"_{prompt_cfg.name}", pay_load=pay_load)
                # Extraction step
                if prompt_cfg.result_extractor:
                    extracted = prompt_cfg.result_extractor(raw)
                    if extracted is None:
                        err_msg = "The response format is incorrect or the content is irrelevant, and the expected result cannot be extracted."
                        self.logger.warning(err_msg)
                        errorMsg_list.append(err_msg)
                        # Provide feedback to LLM for targeted correction
                        current_prompt = self._build_feedback_prompt(prompt=current_prompt, raw=raw, error=err_msg)
                        pay_load.setdefault("validation_error",[]).append(err_msg)
                        last_error = RuntimeError(err_msg)
                        continue
                    result = extracted
                    self.logger.debug(f"Extracted result (truncated): {str(result)}")
                else:
                    result = raw

                # Validation step: validator returns empty string on success, error message otherwise
                if prompt_cfg.result_validator:
                    validation_error = prompt_cfg.result_validator(result)
                    if validation_error and validation_error:
                        self.logger.warning(
                            "Prompt result failed validation; will retry if possible: %s",
                            validation_error,
                        )
                        errorMsg_list.append(validation_error)
                        current_prompt = self._build_feedback_prompt(prompt=current_prompt, raw=raw, error=validation_error)
                        pay_load.setdefault("validation_error",[]).append(validation_error)
                        last_error = ValueError(validation_error)
                        continue

                self.logger.info("Prompt succeeded and passed validation")
                # include last_reason in the result payload for downstream use
                return {"raw": raw, "result": result, "reason": self.last_reason, "error_messages": errorMsg_list}
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Prompt attempt {attempt} failed: {exc}")
                last_error = exc
                errorMsg_list.append(str(exc))
                continue

        self.logger.error("LLM prompt failed after all retries")
        return {"raw": None, "result": None, "reason": self.last_reason, "error": last_error, "error_messages": errorMsg_list}
    
    def _build_feedback_prompt(self, prompt: str, raw: str, error: str) -> str:
        """Append structured error feedback to the original prompt for targeted correction."""
        if self.use_cn:
            feedback_block = (
                "\n\n[Feedback]\n"
                "你的上一次回答存在问题，需要针对性修复。\n"
                f"错误信息: {error}\n"
                "请根据错误信息修改你的输出，并严格遵守原始要求的输出格式。\n"
                "如果需要，请仅返回修正后的内容，不要附加额外描述。\n"
            )
            # Optionally provide a truncated preview of raw output to help the model
            preview = f"\n上一次输出片段(截断):\n{raw[:500]}\n"
        else:
            feedback_block = (
                "\n\n[Feedback]\n"
                "There was an issue with your last response that needs to be specifically addressed.\n"
                f"Error details: {error}\n"
                "Please modify your output according to the error message and strictly adhere to the original output format requirements.\n"
                "If necessary, return only the corrected content without any additional descriptions.\n"
            )
            # Optionally provide a truncated preview of raw output to help the model
            preview = f"\nLast output snippet (truncated):\n{raw[:500]}\n"
        return prompt + feedback_block + preview

    def _build_cache_key(
        self, pay_load
    ) -> str:
        """Create a deterministic cache key from request parameters."""
        fingerprint = [[k,v] for k, v in sorted(pay_load.items())]
        fingerprint_str = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True)
        self.logger.info(f"fingerprint_str: {fingerprint_str}")
        digest = hashlib.sha256(fingerprint_str.encode("utf-8")).hexdigest()
        return digest

    def _cache_path(self, cache_key: str, sub_dir: str) -> Path:
        """Map cache key to a cache file path."""
        dir = Path(llm_cache_path) / sub_dir
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
        return Path(llm_cache_path) / sub_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str, sub_dir: str) -> Optional[Dict[str, Any]]:
        """Read cached response if available."""
        path = self._cache_path(cache_key, sub_dir)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"Failed to read cache {path}: {exc}")
            return None

    def _write_cache(self, cache_key: str, data: Dict[str, Any], sub_dir: str) -> None:
        """Persist response cache to disk."""
        path = self._cache_path(cache_key, sub_dir)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"Failed to write cache {path}: {exc}")


llm_client = LLMClient()  # Singleton instance for module-level use

def run_prompt_config(
        prompt_cfg, 
        sub_dir="default",
        use_cn=False,
        format_kwargs={}
):
    """Convenience function to run a PromptConfig with the singleton LLMClient."""
    return llm_client.run_prompt_config(
        prompt_cfg=prompt_cfg,
        sub_dir=sub_dir,
        use_cn=use_cn,
        format_kwargs=format_kwargs)
