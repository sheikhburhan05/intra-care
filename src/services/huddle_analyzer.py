"""Service for medication/lab-to-diagnosis gap detection via LLM."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Type

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
# from langchain_anthropic import ChatAnthropic  # Anthropic Sonnet kept as commented reference.
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..config import DEFAULT_HUDDLE_MODEL, FALLBACK_HUDDLE_MODEL, resolve_repo_path
from ..domain.huddle_output import CombinedLabGap, LabGap, MedicationGap
from ..prompts import (
    PROMPT_COMBINED_REPORT_GAP_ANALYSIS,
    PROMPT_MEDICATION_GAP_ANALYSIS,
    PROMPT_SINGLE_REPORT_GAP_ANALYSIS,
    PROMPT_THRESHOLD_SEARCH,
    THRESHOLDS_BLOCK_FALLBACK,
    THRESHOLDS_BLOCK_WITH_RESULTS,
)
from ..repositories.patient_repository import PatientRepository
from .huddle_pdf_exporter import HuddlePdfExporter


class SingleReportLabOutput(BaseModel):
    suspected_gaps: list[LabGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Summary of missing diagnosis candidates for this report")


class MedicationAnalysisOutput(BaseModel):
    suspected_gaps: list[MedicationGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Summary of medication-problem list gaps")


class CombinedLabAnalysisOutput(BaseModel):
    suspected_gaps: list[CombinedLabGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Summary of combined multi-report diagnosis gaps")


class HuddleAnalyzer:
    """Run structured huddle analysis for a single patient."""

    def __init__(self, repository: Optional[PatientRepository] = None) -> None:
        self.repository = repository or PatientRepository()
        self.pdf_exporter = HuddlePdfExporter()
        self._ensure_provider_env()

    def analyze_patient_huddle(
        self,
        patient_id: str,
        patients_json_path: str = "patients.json",
        output_dir: Optional[str] = None,
        model: str = DEFAULT_HUDDLE_MODEL,
        use_web_search: bool = True,
        use_llm_tools: bool = True,
        enable_combined_lab_analysis: bool = True,
    ) -> dict[str, Any]:
        json_path = resolve_repo_path(patients_json_path)
        out_dir = Path(output_dir) if output_dir else json_path.parent
        if not out_dir.is_absolute():
            out_dir = resolve_repo_path(out_dir)

        patients = self.repository.load_patients_from_json(json_path)
        pid_key = str(patient_id)
        if pid_key not in patients:
            raise ValueError(f"Patient {patient_id} not found in {json_path}")

        patient = patients[pid_key]
        problem_list = self._extract_problems(patient)
        medications = self._extract_medications(patient)
        lab_reports = patient.get("lab_reports", [])
        labs = self._extract_labs(patient)
        if use_web_search:
            print("LLM using search tool to fetch clinical lab thresholds...")
        thresholds_context = (
            self._fetch_clinical_thresholds(labs, model=model, use_llm_tools=use_llm_tools) if use_web_search else ""
        )
        thresholds_block = (
            THRESHOLDS_BLOCK_WITH_RESULTS.format(thresholds_context=thresholds_context)
            if thresholds_context
            else THRESHOLDS_BLOCK_FALLBACK
        )
        llm = self._build_llm(model)
        fallback_llm = self._build_llm(FALLBACK_HUDDLE_MODEL)
        report_llm = llm.with_structured_output(SingleReportLabOutput)
        fallback_report_llm = fallback_llm.with_structured_output(SingleReportLabOutput)
        medication_llm = llm.with_structured_output(MedicationAnalysisOutput)
        fallback_medication_llm = fallback_llm.with_structured_output(MedicationAnalysisOutput)
        combined_llm = llm.with_structured_output(CombinedLabAnalysisOutput)
        fallback_combined_llm = fallback_llm.with_structured_output(CombinedLabAnalysisOutput)

        medication_prompt = PROMPT_MEDICATION_GAP_ANALYSIS.format(
            problem_list_json=json.dumps(problem_list, indent=2),
            medications_json=json.dumps(medications, indent=2),
        )
        medication_result = self._invoke_with_fallback(
            primary=medication_llm,
            fallback=fallback_medication_llm,
            fallback_base_llm=fallback_llm,
            messages=[HumanMessage(content=medication_prompt)],
            operation_name="medication gap analysis",
            schema_cls=MedicationAnalysisOutput,
        )
        medication_gaps = [gap.model_dump() for gap in medication_result.suspected_gaps if not gap.in_problem_list]

        all_report_gaps: list[dict[str, Any]] = []
        report_summaries: list[str] = []
        for report in lab_reports:
            report_id = str(report.get("lab_report_id", "")).strip()
            report_results = self._format_report_results(report)
            if not report_results:
                continue
            report_prompt = PROMPT_SINGLE_REPORT_GAP_ANALYSIS.format(
                thresholds_block=thresholds_block,
                problem_list_json=json.dumps(problem_list, indent=2),
                report_id=report_id or "unknown",
                report_results=report_results,
            )
            report_result = self._invoke_with_fallback(
                primary=report_llm,
                fallback=fallback_report_llm,
                fallback_base_llm=fallback_llm,
                messages=[HumanMessage(content=report_prompt)],
                operation_name=f"per-report lab analysis [{report_id or 'unknown'}]",
                schema_cls=SingleReportLabOutput,
            )
            report_summaries.append(f"{report_id or 'unknown'}: {report_result.summary}")
            for gap in report_result.suspected_gaps:
                gap_dict = gap.model_dump()
                gap_dict["lab_report_id"] = report_id or gap_dict.get("lab_report_id", "")
                gap_dict["in_problem_list"] = bool(gap_dict.get("in_problem_list", False))
                if gap_dict["in_problem_list"]:
                    continue
                all_report_gaps.append(gap_dict)

        deduped_report_gaps = self._dedupe_lab_gaps(all_report_gaps)
        lab_summary = self._build_lab_summary(deduped_report_gaps, report_summaries)
        combined_gaps, combined_summary = self._run_combined_lab_analysis(
            combined_llm=combined_llm,
            fallback_combined_llm=fallback_combined_llm,
            enable_combined_lab_analysis=enable_combined_lab_analysis,
            lab_reports=lab_reports,
            problem_list=problem_list,
            deduped_report_gaps=deduped_report_gaps,
            thresholds_block=thresholds_block,
        )
        summary_note = self._build_summary_note(
            problem_list=problem_list,
            medications=medications,
            medication_gaps=medication_gaps,
            deduped_report_gaps=deduped_report_gaps,
            combined_gaps=combined_gaps,
        )

        output = {
            "patient_id": patient_id,
            "medication_to_diagnosis": {
                "suspected_gaps": medication_gaps,
                "summary": medication_result.summary
                or self._build_medication_summary(medication_gaps),
            },
            "lab_report_to_diagnosis": {
                "suspected_gaps": deduped_report_gaps,
                "summary": lab_summary,
            },
            "combined_lab_report_to_diagnosis": {
                "suspected_gaps": combined_gaps,
                "summary": combined_summary,
            },
            "summary_note_before_huddle": summary_note,
        }

        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / f"{pid_key}.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)
        print(f"Saved huddle analysis to {out_path}")

        pdf_path = out_dir / f"{pid_key}.pdf"
        self.pdf_exporter.export(patient_id=pid_key, analysis=output, output_path=pdf_path)
        print(f"Saved huddle PDF to {pdf_path}")
        return output

    @staticmethod
    def _format_report_results(report: dict[str, Any]) -> str:
        lines: list[str] = []
        for result in report.get("results", []):
            analyte = str(result.get("labanalyte", "")).strip()
            value = str(result.get("labvalue", "")).strip()
            if analyte:
                lines.append(f"- {analyte}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _dedupe_lab_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for gap in gaps:
            key = (
                str(gap.get("lab_report_id", "")).strip().lower(),
                str(gap.get("lab_analyte", "")).strip().lower(),
                str(gap.get("lab_value", "")).strip().lower(),
                str(gap.get("implied_condition", "")).strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(gap)
        return deduped

    @staticmethod
    def _build_lab_summary(deduped_report_gaps: list[dict[str, Any]], report_summaries: list[str]) -> str:
        if not deduped_report_gaps:
            return "No clear missing diagnosis candidates found from per-report lab analysis."
        report_count = len({g.get("lab_report_id", "") for g in deduped_report_gaps})
        gap_count = len(deduped_report_gaps)
        return (
            f"Detected {gap_count} missing diagnosis candidate(s) "
            f"across {report_count} lab report(s). "
            f"Report notes: {' | '.join(report_summaries[:5])}"
        )

    @staticmethod
    def _build_medication_summary(medication_gaps: list[dict[str, Any]]) -> str:
        if not medication_gaps:
            return "No clear medication-to-diagnosis mismatches identified."
        return f"Detected {len(medication_gaps)} medication item(s) requiring diagnosis linkage review."

    @staticmethod
    def _build_combined_lab_summary(combined_gaps: list[dict[str, Any]]) -> str:
        if not combined_gaps:
            return "No additional combined multi-report diagnosis gaps identified."
        return f"Detected {len(combined_gaps)} combined multi-report diagnosis gap(s)."

    def _run_combined_lab_analysis(
        self,
        combined_llm: Any,
        fallback_combined_llm: Any,
        enable_combined_lab_analysis: bool,
        lab_reports: list[dict[str, Any]],
        problem_list: list[str],
        deduped_report_gaps: list[dict[str, Any]],
        thresholds_block: str,
    ) -> tuple[list[dict[str, Any]], str]:
        if not enable_combined_lab_analysis:
            return [], "Combined multi-report analysis disabled by flag."
        if len(lab_reports) < 2 and len(deduped_report_gaps) < 2:
            return [], "Insufficient multi-report context for combined analysis."

        prompt = PROMPT_COMBINED_REPORT_GAP_ANALYSIS.format(
            thresholds_block=thresholds_block,
            problem_list_json=json.dumps(problem_list, indent=2),
            per_report_gaps_json=json.dumps(deduped_report_gaps, indent=2),
            all_reports_snapshot=self._format_all_reports_for_combined(lab_reports),
        )
        result = self._invoke_with_fallback(
            primary=combined_llm,
            fallback=fallback_combined_llm,
            fallback_base_llm=self._build_llm(FALLBACK_HUDDLE_MODEL),
            messages=[HumanMessage(content=prompt)],
            operation_name="combined multi-report lab analysis",
            schema_cls=CombinedLabAnalysisOutput,
        )
        combined_gaps = []
        for gap in result.suspected_gaps:
            gap_dict = gap.model_dump()
            gap_dict["contributing_report_ids"] = [
                str(report_id).strip() for report_id in gap_dict.get("contributing_report_ids", []) if str(report_id).strip()
            ]
            gap_dict["in_problem_list"] = bool(gap_dict.get("in_problem_list", False))
            if gap_dict["in_problem_list"]:
                continue
            combined_gaps.append(gap_dict)
        deduped = self._dedupe_combined_lab_gaps(combined_gaps)
        return deduped, (result.summary or self._build_combined_lab_summary(deduped))

    @staticmethod
    def _format_all_reports_for_combined(lab_reports: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for report in lab_reports:
            report_id = str(report.get("lab_report_id", "")).strip() or "unknown"
            lines = [f"Report ID: {report_id}"]
            for result in report.get("results", [])[:40]:
                analyte = str(result.get("labanalyte", "")).strip()
                value = str(result.get("labvalue", "")).strip()
                if analyte:
                    lines.append(f"- {analyte}: {value}")
            chunks.append("\n".join(lines))
        return "\n\n".join(chunks[:50])

    @staticmethod
    def _dedupe_combined_lab_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        deduped: list[dict[str, Any]] = []
        for gap in gaps:
            report_ids = tuple(sorted(set(gap.get("contributing_report_ids", []))))
            key = (str(gap.get("implied_condition", "")).strip().lower(), report_ids)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(gap)
        return deduped

    @staticmethod
    def _build_summary_note(
        problem_list: list[str],
        medications: list[str],
        medication_gaps: list[dict[str, Any]],
        deduped_report_gaps: list[dict[str, Any]],
        combined_gaps: list[dict[str, Any]],
    ) -> dict[str, str]:
        if not deduped_report_gaps and not medication_gaps and not combined_gaps:
            return {
                "context": (
                    "Medication review, per-report lab review, and combined multi-report review found no clear missing diagnosis gaps."
                ),
                "suggested_huddle_note_bullet": "No additional medication- or lab-driven coding gaps identified at this time.",
                "physician_prompt": "Continue routine monitoring and update coding if clinically significant changes emerge.",
            }
        med_conditions = ", ".join(
            list(
                dict.fromkeys(
                    str(gap.get("implied_condition", "")).strip()
                    for gap in medication_gaps
                    if str(gap.get("implied_condition", "")).strip()
                )
            )[:5]
        )
        lab_conditions = ", ".join(
            list(
                dict.fromkeys(
                    str(gap.get("implied_condition", "")).strip()
                    for gap in deduped_report_gaps
                    if str(gap.get("implied_condition", "")).strip()
                )
            )[:5]
        )
        combined_conditions = ", ".join(
            list(
                dict.fromkeys(
                    str(gap.get("implied_condition", "")).strip()
                    for gap in combined_gaps
                    if str(gap.get("implied_condition", "")).strip()
                )
            )[:5]
        )
        return {
            "context": (
                f"Problem list reviewed ({len(problem_list)} entries), medications reviewed ({len(medications)} active), "
                f"medication gaps detected ({len(medication_gaps)}), and lab-report gaps detected ({len(deduped_report_gaps)}). "
                f"Medication-implied conditions: {med_conditions or 'none'}. "
                f"Lab-implied conditions: {lab_conditions or 'none'}. "
                f"Combined multi-report conditions: {combined_conditions or 'none'}."
            ),
            "suggested_huddle_note_bullet": (
                "Review flagged medications, report-level lab findings, and combined multi-report signals; then confirm coding updates."
            ),
            "physician_prompt": (
                "For each medication, report-level lab flag, and combined trend signal, confirm diagnosis relevance/status and update problem list/coding where appropriate."
            ),
        }

    def _fetch_clinical_thresholds(self, labs: list[dict[str, str]], model: str, use_llm_tools: bool = True) -> str:
        if use_llm_tools:
            try:
                return self._fetch_clinical_thresholds_with_llm_tools(labs, model)
            except Exception as exc:
                if self._is_tool_validation_error(exc):
                    print("LLM tool validation failed (model attempted invalid tool call). Falling back to direct search.")
                    return self._fetch_clinical_thresholds_direct(labs)
                raise
        return self._fetch_clinical_thresholds_direct(labs)

    def _fetch_clinical_thresholds_direct(self, labs: list[dict[str, str]]) -> str:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun

            search = DuckDuckGoSearchRun()
            queries = [
                "clinical lab reference ranges eGFR A1C BNP creatinine hemoglobin 2024",
                "ICD-10 lab value thresholds diabetes CKD heart failure anemia",
            ]
            chunks: list[str] = []
            for query in queries:
                try:
                    result = search.invoke(query)
                    self._log_debug(
                        "tool.ddg.query",
                        {"input": {"query": query}, "output": {"output_preview": str(result)[:600]}},
                    )
                    if result and len(result) > 50:
                        chunks.append(f"Search: {query}\n{result[:2000]}")
                except Exception:
                    continue
            if chunks:
                return "\n\n---\n\n".join(chunks) + "\n\nUse the above retrieved guidelines where applicable."
            return ""
        except ImportError:
            return ""
        except Exception:
            return ""

    def _fetch_clinical_thresholds_with_llm_tools(self, labs: list[dict[str, str]], model: str) -> str:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            from langchain_core.messages import ToolMessage

            search_tool = DuckDuckGoSearchRun()
            llm = self._build_llm(model)
            fallback_llm = self._build_llm(FALLBACK_HUDDLE_MODEL)
            llm_with_tools = llm.bind_tools([search_tool])
            fallback_llm_with_tools = fallback_llm.bind_tools([search_tool])

            lab_names = list(dict.fromkeys(lab.get("labanalyte", "") for lab in labs if lab.get("labanalyte")))[:30]
            lab_preview = ", ".join(lab_names) if lab_names else "various lab tests"
            prompt = PROMPT_THRESHOLD_SEARCH.format(lab_preview=lab_preview)

            messages = [HumanMessage(content=prompt)]
            all_results: list[str] = []
            for _ in range(5):
                response = self._invoke_with_fallback(
                    primary=llm_with_tools,
                    fallback=fallback_llm_with_tools,
                    fallback_base_llm=fallback_llm,
                    messages=messages,
                    operation_name="threshold search planning",
                )
                if not getattr(response, "tool_calls", None):
                    break
                tool_messages = []
                for index, tool_call in enumerate(response.tool_calls):
                    args = tool_call.get("args") or {}
                    query = args.get("query") or args.get("input")
                    if isinstance(query, dict):
                        query = query.get("query") or query.get("input") or ""
                    query = str(query) if query else ""
                    try:
                        output = search_tool.invoke(query)[:1500] if query else "[No query]"
                        all_results.append(f"Query: {query}\n{output}")
                        self._log_debug(
                            "tool.ddg.tool_call",
                            {"input": {"query": query}, "output": {"output_preview": output[:600]}},
                        )
                    except Exception:
                        output = "[Search failed]"
                        all_results.append(f"Query: {query}\n[Search failed]")
                        self._log_debug(
                            "tool.ddg.tool_call_error",
                            {"input": {"query": query}, "output": {"error": "[Search failed]"}},
                        )
                    tool_messages.append(ToolMessage(content=output, tool_call_id=tool_call.get("id", str(index))))
                messages.append(response)
                messages.extend(tool_messages)

            if all_results:
                return "\n\n---\n\n".join(all_results) + "\n\nUse the above retrieved guidelines where applicable."
            return ""
        except ImportError:
            return ""
        except Exception as exc:
            if self._is_tool_validation_error(exc):
                raise  # Let caller fall back to direct search
            return ""

    @staticmethod
    def _build_llm(model: str) -> Any:
        # Anthropic reference (disabled):
        # return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
        if model.startswith("gemini-"):
            return ChatGoogleGenerativeAI(model=model, temperature=0)
        return ChatGroq(model=model, temperature=0)

    def _invoke_with_fallback(
        self,
        primary: Any,
        fallback: Any,
        fallback_base_llm: Any,
        messages: list[Any],
        operation_name: str,
        schema_cls: Optional[Type[BaseModel]] = None,
    ) -> Any:
        try:
            return primary.invoke(messages)
        except Exception as exc:
            if schema_cls and self._is_tool_validation_error(exc):
                repaired = self._try_parse_failed_generation(exc, schema_cls)
                if repaired is not None:
                    print(f"Recovered structured output from tool_use_failed for {operation_name}.")
                    return repaired
                print(
                    f"Could not parse failed_generation for {operation_name}. "
                    "Retrying with strict JSON repair."
                )
                return self._repair_structured_output_with_base_llm(
                    base_llm=fallback_base_llm,
                    messages=messages,
                    schema_cls=schema_cls,
                )
            if self._is_limit_error(exc):
                print(
                    f"Primary Groq model '{DEFAULT_HUDDLE_MODEL}' hit a limit during {operation_name}. "
                    f"Retrying with fallback '{FALLBACK_HUDDLE_MODEL}'."
                )
                try:
                    return fallback.invoke(messages)
                except Exception as fallback_exc:
                    if schema_cls and self._is_output_parse_error(fallback_exc):
                        print(
                            f"Fallback model '{FALLBACK_HUDDLE_MODEL}' returned non-JSON for {operation_name}. "
                            "Retrying with strict JSON repair."
                        )
                        return self._repair_structured_output_with_base_llm(
                            base_llm=fallback_base_llm,
                            messages=messages,
                            schema_cls=schema_cls,
                        )
                    raise
            raise

    def _try_parse_failed_generation(
        self, exc: Exception, schema_cls: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Extract and parse failed_generation from Groq tool_use_failed error."""
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
    def _is_tool_validation_error(exc: Exception) -> bool:
        """Detect when the model attempted to call a tool not in the request (e.g. 'json')."""
        text = str(exc).lower()
        return "tool_use_failed" in text or "which was not in request.tools" in text or "attempted to call tool" in text

    @staticmethod
    def _is_limit_error(exc: Exception) -> bool:
        text = str(exc).lower()
        limit_markers = [
            "rate limit",
            "429",
            "quota",
            "tokens per minute",
            "request too large",
            "context length",
            "limit exceeded",
        ]
        return any(marker in text for marker in limit_markers)

    @staticmethod
    def _is_output_parse_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = [
            "output_parse_failed",
            "parsing failed",
            "could not be parsed",
            "invalid json",
        ]
        return any(marker in text for marker in markers)

    def _repair_structured_output_with_base_llm(
        self,
        base_llm: Any,
        messages: list[Any],
        schema_cls: Type[BaseModel],
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
        repaired = base_llm.invoke([*messages, repair_instruction])
        content = getattr(repaired, "content", "")
        parsed_obj = self._parse_json_object_from_text(str(content))
        return schema_cls.model_validate(parsed_obj)

    @staticmethod
    def _parse_json_object_from_text(text: str) -> dict[str, Any]:
        text = text.strip()
        # Try direct JSON first.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Fallback: extract first top-level JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Unable to parse JSON object from model output.")

    @staticmethod
    def _ensure_provider_env() -> None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = gemini_api_key

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
            return {str(k): HuddleAnalyzer._serialize_debug_payload(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [HuddleAnalyzer._serialize_debug_payload(v) for v in payload]
        if isinstance(payload, (str, int, float, bool)) or payload is None:
            return payload
        return str(payload)

    @staticmethod
    def _extract_medications(patient: dict[str, Any]) -> list[str]:
        medications: list[str] = []
        for medication in patient.get("medications", []):
            names = medication.get("med names") or medication.get("med_names") or ""
            for part in re.split(r"[;,]+\s*", str(names)):
                part = part.strip()
                if part:
                    medications.append(part)
        return list(dict.fromkeys(medications))

    @staticmethod
    def _extract_problems(patient: dict[str, Any]) -> list[str]:
        return [
            (problem.get("patientsnomedproblemdesc") or problem.get("problem_desc") or "").strip()
            for problem in patient.get("problems", [])
            if (problem.get("patientsnomedproblemdesc") or problem.get("problem_desc"))
        ]

    @staticmethod
    def _extract_labs(patient: dict[str, Any]) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        labs: list[dict[str, str]] = []
        for report in patient.get("lab_reports", []):
            for result in report.get("results", []):
                analyte = (result.get("labanalyte") or "").strip()
                value = (result.get("labvalue") or "").strip()
                if analyte and (analyte, value) not in seen:
                    seen.add((analyte, value))
                    labs.append({"labanalyte": analyte, "labvalue": value})
        return labs
