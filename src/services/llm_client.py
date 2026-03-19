"""LLM client: model building, invocation with fallback, JSON repair, and logging."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel

from ..config import DEFAULT_HUDDLE_MODEL, FALLBACK_HUDDLE_MODEL


class LLMClient:
    """Thin wrapper around LangChain LLMs: building, invoking, repairing, and logging."""

    def __init__(self) -> None:
        self._ensure_provider_env()

    # ── Model factory ─────────────────────────────────────────────────────────

    @staticmethod
    def build(model: str) -> Any:
        """Instantiate the appropriate LangChain chat model for *model*."""
        if model.startswith("claude-"):
            return ChatAnthropic(model=model, temperature=0)
        if model.startswith("gemini-"):
            return ChatGoogleGenerativeAI(model=model, temperature=0)
        return ChatGroq(model=model, temperature=0)

    # ── Invocation ─────────────────────────────────────────────────────────────

    def invoke_with_fallback(
        self,
        primary: Any,
        fallback: Any,
        fallback_base_llm: Any,
        messages: list[Any],
        operation_name: str,
        schema_cls: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """Invoke *primary*; on known errors attempt structured repair or *fallback*."""
        self._log_llm_call(operation_name, messages)
        try:
            response = primary.invoke(messages)
            self._log_llm_response(operation_name, response)
            return response
        except Exception as exc:
            if schema_cls and self._is_tool_validation_error(exc):
                repaired = self._try_parse_failed_generation(exc, schema_cls)
                if repaired is not None:
                    print(f"Recovered structured output from tool_use_failed for {operation_name}.")
                    self._log_llm_response(operation_name, repaired, note="recovered from failed_generation")
                    return repaired
                print(
                    f"Could not parse failed_generation for {operation_name}. "
                    "Retrying with strict JSON repair."
                )
                return self._repair_structured_output_with_base_llm(
                    base_llm=fallback_base_llm,
                    messages=messages,
                    schema_cls=schema_cls,
                    operation_name=operation_name,
                )
            if self._is_limit_error(exc):
                print(
                    f"Primary model '{DEFAULT_HUDDLE_MODEL}' hit a limit during {operation_name}. "
                    f"Retrying with fallback '{FALLBACK_HUDDLE_MODEL}'."
                )
                try:
                    self._log_llm_call(operation_name, messages, note="fallback model")
                    response = fallback.invoke(messages)
                    self._log_llm_response(operation_name, response, note="fallback model")
                    return response
                except Exception as fallback_exc:
                    if schema_cls and self._is_output_parse_error(fallback_exc):
                        print(
                            f"Fallback model '{FALLBACK_HUDDLE_MODEL}' returned non-JSON for "
                            f"{operation_name}. Retrying with strict JSON repair."
                        )
                        return self._repair_structured_output_with_base_llm(
                            base_llm=fallback_base_llm,
                            messages=messages,
                            schema_cls=schema_cls,
                            operation_name=operation_name,
                        )
                    raise
            raise

    # ── JSON repair ────────────────────────────────────────────────────────────

    def _repair_structured_output_with_base_llm(
        self,
        base_llm: Any,
        messages: list[Any],
        schema_cls: Type[BaseModel],
        operation_name: str = "json repair",
    ) -> BaseModel:
        schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
        repair_instruction = HumanMessage(
            content=(
                "Your previous answer was not valid JSON.\n"
                "Return ONLY one valid JSON object that strictly conforms to this JSON schema.\n"
                "No markdown. No explanations. No extra text.\n"
                f"{schema_json}"
            )
        )
        repair_messages = [*messages, repair_instruction]
        self._log_llm_call(operation_name, repair_messages, note="JSON repair")
        repaired = base_llm.invoke(repair_messages)
        self._log_llm_response(operation_name, repaired, note="JSON repair")
        content = getattr(repaired, "content", "")
        parsed_obj = self._parse_json_object_from_text(str(content))
        return schema_cls.model_validate(parsed_obj)

    def _try_parse_failed_generation(
        self, exc: Exception, schema_cls: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Extract and parse failed_generation from a Groq tool_use_failed error."""
        text = str(exc)
        start = text.find("'failed_generation'")
        if start == -1:
            start = text.find('"failed_generation"')
        if start == -1:
            return None
        bracket = text.find("[", start)
        if bracket == -1:
            bracket = text.find("{", start)
        if bracket == -1:
            return None
        depth = 0
        in_str = False
        escape = False
        end = bracket
        for i, c in enumerate(text[bracket:], bracket):
            if escape:
                escape = False
                continue
            if c == "\\" and in_str:
                escape = True
                continue
            if in_str:
                if c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
                continue
            if c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            raw = text[bracket:end].replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'")
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == 1:
                parsed = parsed[0]
            if isinstance(parsed, dict):
                return schema_cls.model_validate(parsed)
        except (json.JSONDecodeError, Exception):
            pass
        return None

    @staticmethod
    def _parse_json_object_from_text(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start: end + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Unable to parse JSON object from model output.")

    # ── Error classification ───────────────────────────────────────────────────

    @staticmethod
    def _is_tool_validation_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "tool_use_failed" in text
            or "which was not in request.tools" in text
            or "attempted to call tool" in text
        )

    @staticmethod
    def _is_limit_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            m in text
            for m in [
                "rate limit", "429", "quota", "tokens per minute",
                "request too large", "context length", "limit exceeded",
            ]
        )

    @staticmethod
    def _is_output_parse_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            m in text
            for m in ["output_parse_failed", "parsing failed", "could not be parsed", "invalid json"]
        )

    # ── Environment setup ──────────────────────────────────────────────────────

    @staticmethod
    def _ensure_provider_env() -> None:
        # langchain-google-genai reads GOOGLE_API_KEY; map from GEMINI_API_KEY if set
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        # langchain-anthropic reads ANTHROPIC_API_KEY directly

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_llm_call(
        self, operation_name: str, messages: list[Any], note: str = ""
    ) -> None:
        sep = "-" * 60
        label = f"[LLM REQUEST] {operation_name}" + (f"  ({note})" if note else "")
        prompt_text = self._format_messages_for_log(messages)
        print(f"\n{sep}\n{label}\n{sep}")
        print(prompt_text[:12000])
        print(sep)

    def _log_llm_response(
        self, operation_name: str, response: Any, note: str = ""
    ) -> None:
        pass  # uncomment below to enable response logging
        # sep = "-" * 60
        # label = f"[LLM RESPONSE] {operation_name}" + (f"  ({note})" if note else "")
        # response_text = self._format_response_for_log(response)
        # print(f"\n{sep}\n{label}\n{sep}")
        # print(response_text[:12000])
        # print(sep)

    def _log_debug(self, step: str, payload: Any) -> None:
        separator = "-" * 40
        serialized = self._serialize_debug_payload(payload)
        if isinstance(serialized, dict) and ("input" in serialized or "output" in serialized):
            input_payload = serialized.get("input")
            output_payload = serialized.get("output")
            meta_payload = {k: v for k, v in serialized.items() if k not in {"input", "output"}}
        else:
            input_payload = serialized
            output_payload = None
            meta_payload = None

        print(f"\n{separator}")
        print(f"[DEBUG] {step}")
        if meta_payload:
            print("Meta:")
            print(self._pretty_debug(meta_payload))
        print("Input:")
        print(self._pretty_debug(input_payload))
        print("Output:")
        print(self._pretty_debug(output_payload))
        print(separator)

    @staticmethod
    def _format_messages_for_log(messages: list[Any]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = type(msg).__name__.replace("Message", "").upper()
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(block.get("text", block) if isinstance(block, dict) else block)
                    for block in content
                )
            parts.append(f"[{role}]\n{str(content)}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_response_for_log(response: Any) -> str:
        if response is None:
            return "N/A"
        if hasattr(response, "model_dump"):
            try:
                return json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
            except Exception:
                pass
        content = getattr(response, "content", None)
        if content is not None:
            return str(content)
        try:
            return json.dumps(response, indent=2, ensure_ascii=False)
        except Exception:
            return str(response)

    @staticmethod
    def _pretty_debug(payload: Any) -> str:
        if payload is None:
            return "N/A"
        if isinstance(payload, str):
            return payload[:8000]
        try:
            return json.dumps(payload, indent=2, ensure_ascii=True)[:8000]
        except Exception:
            return str(payload)[:8000]

    @staticmethod
    def _serialize_debug_payload(payload: Any) -> Any:
        if hasattr(payload, "model_dump"):
            try:
                return payload.model_dump()
            except Exception:
                return str(payload)
        if isinstance(payload, dict):
            return {str(k): LLMClient._serialize_debug_payload(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [LLMClient._serialize_debug_payload(v) for v in payload]
        if isinstance(payload, (str, int, float, bool)) or payload is None:
            return payload
        return str(payload)
