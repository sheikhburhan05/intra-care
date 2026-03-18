"""Service for medication/lab-to-diagnosis gap detection via LLM."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional, Type

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..config import DEFAULT_HUDDLE_MODEL, FALLBACK_HUDDLE_MODEL, output_dir as get_output_dir, resolve_repo_path
from ..domain.huddle_output import CombinedLabGap, LabGap, MedicationGap
from ..prompts import (
    PROMPT_COMBINED_REPORT_GAP_ANALYSIS,
    PROMPT_MEDICATION_GAP_ANALYSIS,
    PROMPT_SINGLE_REPORT_GAP_ANALYSIS,
)
from ..summary_prompts import PROMPT_DOCTOR_SUMMARY, PROMPT_LAB_NARRATIVE_SUMMARY
from ..threshold_prompts import THRESHOLDS_BLOCK_FALLBACK
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
        enable_medication_analysis: bool = True,
        enable_per_report_lab_analysis: bool = True,
        enable_combined_lab_analysis: bool = True,
        enable_doctor_summary: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        json_path = resolve_repo_path(patients_json_path)
        if output_dir:
            out_dir = Path(output_dir) if Path(output_dir).is_absolute() else resolve_repo_path(output_dir)
        else:
            out_dir = get_output_dir()

        patients = self.repository.load_patients_from_json(json_path)
        pid_key = str(patient_id)
        if pid_key not in patients:
            raise ValueError(f"Patient {patient_id} not found in {json_path}")

        patient = patients[pid_key]
        problem_list = self._extract_problems(patient)
        medications = self._extract_medications(patient)
        lab_reports = patient.get("lab_reports", [])

        # Base thresholds block — each report extends this with its own targeted search.
        thresholds_block = THRESHOLDS_BLOCK_FALLBACK

        llm = self._build_llm(model)
        fallback_llm = self._build_llm(FALLBACK_HUDDLE_MODEL)

        def _notify(msg: str) -> None:
            print(msg)
            if progress_callback:
                progress_callback(msg)

        # ── Phase 1 (parallel): medication + all per-report lab analyses ──
        medication_gaps: list[dict[str, Any]] = []
        medication_summary = "Medication analysis skipped."
        deduped_report_gaps: list[dict[str, Any]] = []

        # Build the LLM variants needed for phase 1
        medication_llm = llm.with_structured_output(MedicationAnalysisOutput) if enable_medication_analysis else None
        fallback_medication_llm = fallback_llm.with_structured_output(MedicationAnalysisOutput) if enable_medication_analysis else None
        report_llm = llm.with_structured_output(SingleReportLabOutput) if enable_per_report_lab_analysis else None
        fallback_report_llm = fallback_llm.with_structured_output(SingleReportLabOutput) if enable_per_report_lab_analysis else None

        # Count valid reports upfront for progress display
        valid_reports = [
            r for r in lab_reports if self._format_report_results(r)
        ] if enable_per_report_lab_analysis else []
        total_reports = len(valid_reports)

        # Collect all tasks: one future per analysis unit
        futures: dict[Any, str] = {}  # future → task label
        max_workers = min(1 + len(valid_reports), 12)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:

            # Submit medication analysis
            if enable_medication_analysis:
                medication_prompt = PROMPT_MEDICATION_GAP_ANALYSIS.format(
                    problem_list_json=json.dumps(problem_list, indent=2),
                    medications_json=json.dumps(medications, indent=2),
                )
                futures[pool.submit(
                    self._invoke_with_fallback,
                    primary=medication_llm,
                    fallback=fallback_medication_llm,
                    fallback_base_llm=fallback_llm,
                    messages=[HumanMessage(content=medication_prompt)],
                    operation_name="medication gap analysis",
                    schema_cls=MedicationAnalysisOutput,
                )] = "__medication__"

            # Submit one future per lab report (each thread: search ranges → LLM)
            for report in valid_reports:
                report_id = str(report.get("lab_report_id", "")).strip()
                futures[pool.submit(
                    self._analyze_single_report,
                    report=report,
                    report_id=report_id,
                    problem_list=problem_list,
                    report_llm=report_llm,
                    fallback_report_llm=fallback_report_llm,
                    fallback_llm=fallback_llm,
                    global_thresholds_block=thresholds_block,
                    use_web_search=use_web_search,
                )] = report_id or "unknown"

            # Collect results as they complete — runs in main thread, safe to call progress_callback
            all_report_gaps: list[dict[str, Any]] = []
            reports_done = 0
            for future in as_completed(futures):
                label = futures[future]
                result = future.result()  # re-raises any exception from the thread
                if label == "__medication__":
                    medication_gaps = [gap.model_dump() for gap in result.suspected_gaps]
                    medication_summary = result.summary or self._build_medication_summary(medication_gaps)
                    gap_count = len(medication_gaps)
                    _notify(f"✅ Medication Gap Analysis — {gap_count} gap(s) found")
                else:
                    reports_done += 1
                    for gap in result.suspected_gaps:
                        gap_dict = gap.model_dump()
                        gap_dict["lab_report_id"] = label or gap_dict.get("lab_report_id", "")
                        all_report_gaps.append(gap_dict)
                    short_name = self._short_report_name(label)
                    gap_count = len(result.suspected_gaps)
                    _notify(
                        f"✅ Report {reports_done}/{total_reports}: {short_name}"
                        f" — {gap_count} gap(s) found"
                    )

        if enable_per_report_lab_analysis:
            deduped_report_gaps = self._dedupe_lab_gaps(all_report_gaps)

        # ── Combined multi-report lab analysis ────────────────────────────
        combined_gaps: list[dict[str, Any]] = []
        combined_summary = "Combined analysis skipped."
        if enable_combined_lab_analysis or enable_per_report_lab_analysis:
            _notify("⏳ Running combined multi-report analysis…")
            combined_llm = llm.with_structured_output(CombinedLabAnalysisOutput)
            fallback_combined_llm = fallback_llm.with_structured_output(CombinedLabAnalysisOutput)
            combined_gaps, combined_summary = self._run_combined_lab_analysis(
                combined_llm=combined_llm,
                fallback_combined_llm=fallback_combined_llm,
                enable_combined_lab_analysis=enable_combined_lab_analysis,
                lab_reports=lab_reports,
                problem_list=problem_list,
                deduped_report_gaps=deduped_report_gaps,
                thresholds_block=thresholds_block,
            )
            _notify(f"✅ Combined analysis — {len(combined_gaps)} pattern(s) found")

        # ── Lab narrative summary ─────────────────────────────────────────
        lab_narrative_summary = ""
        if enable_per_report_lab_analysis or enable_combined_lab_analysis:
            _notify("⏳ Generating lab narrative summary…")
            lab_narrative_summary = self._generate_lab_narrative_summary(
                llm=llm,
                fallback_llm=fallback_llm,
                deduped_report_gaps=deduped_report_gaps,
                combined_gaps=combined_gaps,
                problem_list=problem_list,
            )
            _notify("✅ Lab narrative summary complete")

        # ── Doctor pre-huddle summary ─────────────────────────────────────
        doctor_summary = ""
        if enable_doctor_summary:
            _notify("⏳ Generating doctor pre-huddle summary…")
            doctor_summary = self._generate_doctor_summary(
                llm=llm,
                fallback_llm=fallback_llm,
                medication_gaps=medication_gaps,
                deduped_report_gaps=deduped_report_gaps,
                combined_gaps=combined_gaps,
            )
            _notify("✅ Doctor pre-huddle summary complete")

        output = {
            "patient_id": patient_id,
            "medication_to_diagnosis": {
                "suspected_gaps": medication_gaps,
                "summary": medication_summary,
            },
            "lab_report_to_diagnosis": {
                "suspected_gaps": deduped_report_gaps,
                "narrative_summary": lab_narrative_summary,
            },
            "combined_lab_report_to_diagnosis": {
                "suspected_gaps": combined_gaps,
                "summary": combined_summary,
            },
            "doctor_summary": doctor_summary,
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
    def _short_report_name(report_id: str, max_len: int = 40) -> str:
        """Return a concise display name from a raw lab report ID string.

        IDs often look like '005009    CBC WITH DIFFERENTIAL/PLATELET'.
        We strip the leading code and return only the descriptive part.
        """
        parts = re.split(r"\s{2,}", report_id.strip(), maxsplit=1)
        name = parts[-1] if len(parts) > 1 else report_id
        return name[:max_len] + ("…" if len(name) > max_len else "")

    def _fetch_report_thresholds(self, report: dict[str, Any]) -> str:
        """Search DuckDuckGo for reference ranges of the specific analytes in this report.

        Runs inside a worker thread — results are appended to the per-report prompt.
        """
        report_id = str(report.get("lab_report_id", "unknown")).strip()
        print(f"\n[DDG] _fetch_report_thresholds called for report_id={report_id!r}")
        try:
            print("[DDG] Importing DuckDuckGoSearchRun …")
            from langchain_community.tools import DuckDuckGoSearchRun
            print("[DDG] Import OK")

            raw_results = report.get("results", [])
            print(f"[DDG] report has {len(raw_results)} result rows")

            analytes = [
                str(r.get("labanalyte", "")).strip()
                for r in raw_results
                if r.get("labanalyte")
            ][:10]

            print(f"[DDG] analytes extracted ({len(analytes)}): {analytes}")

            if not analytes:
                print("[DDG] No analytes found — skipping search, returning empty string")
                return ""

            report_name = self._short_report_name(
                str(report.get("lab_report_id", "")).strip(), max_len=60
            )
            analyte_str = ", ".join(analytes[:6])
            query = f"{report_name} {analyte_str} clinical reference range normal values"
            print(f"[DDG] Search query: {query!r}")

            search = DuckDuckGoSearchRun()
            self._log_debug(
                "tool.ddg.per_report_threshold",
                {"input": {"query": query}, "output": {}},
            )
            print("[DDG] Invoking DuckDuckGo search …")
            result = search.invoke(query)
            print(f"[DDG] Search returned {len(result) if result else 0} chars")

            if result and len(result) > 50:
                self._log_debug(
                    "tool.ddg.per_report_threshold",
                    {"input": {"query": query}, "output": {"output_preview": result[:600]}},
                )
                header = f"Reference ranges retrieved for: {analyte_str}"
                print(f"[DDG] Success — returning {len(result[:2000])} chars of context")
                return f"{header}\n{result[:2000]}"

            print("[DDG] Result too short or empty — returning empty string")
            return ""
        except Exception as exc:  # noqa: BLE001
            print(f"[DDG] Exception in _fetch_report_thresholds: {type(exc).__name__}: {exc}")
            return ""

    def _analyze_single_report(
        self,
        report: dict[str, Any],
        report_id: str,
        problem_list: list[str],
        report_llm: Any,
        fallback_report_llm: Any,
        fallback_llm: Any,
        global_thresholds_block: str,
        use_web_search: bool,
    ) -> Any:
        """Fetch targeted reference ranges for this report then invoke the LLM.

        Designed to run inside a ThreadPoolExecutor worker thread.
        """
        report_results = self._format_report_results(report)

        # Build the thresholds block: global + report-specific search results
        thresholds_block = global_thresholds_block
        print(f"\n[DDG] _analyze_single_report: report_id={report_id!r}, use_web_search={use_web_search}")
        if use_web_search:
            print(f"[DDG] Web search enabled — calling _fetch_report_thresholds for {report_id!r}")
            report_ranges = self._fetch_report_thresholds(report)
            if report_ranges:
                print(f"[DDG] Appending {len(report_ranges)} chars of report-specific ranges to prompt")
                thresholds_block = (
                    global_thresholds_block.rstrip()
                    + f"\n\n## Report-Specific Reference Ranges (retrieved via search)\n{report_ranges}"
                )
            else:
                print(f"[DDG] No report-specific ranges returned — using base thresholds only")
        else:
            print("[DDG] Web search disabled — using base thresholds only")

        report_prompt = PROMPT_SINGLE_REPORT_GAP_ANALYSIS.format(
            thresholds_block=thresholds_block,
            problem_list_json=json.dumps(problem_list, indent=2),
            report_id=report_id or "unknown",
            report_results=report_results,
        )
        return self._invoke_with_fallback(
            primary=report_llm,
            fallback=fallback_report_llm,
            fallback_base_llm=fallback_llm,
            messages=[HumanMessage(content=report_prompt)],
            operation_name=f"per-report lab analysis [{report_id or 'unknown'}]",
            schema_cls=SingleReportLabOutput,
        )

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
    def _build_medication_summary(medication_gaps: list[dict[str, Any]]) -> str:
        if not medication_gaps:
            return "No medication-diagnosis gaps identified."
        parts = [
            f"{gap.get('medication', 'Unknown')} implies {gap.get('implied_condition', 'unknown condition')}"
            f" ({gap.get('icd10_code', 'UNKNOWN')}) not present in problem list"
            for gap in medication_gaps
        ]
        return "; ".join(parts) + "."

    @staticmethod
    def _build_combined_lab_summary(combined_gaps: list[dict[str, Any]]) -> str:
        if not combined_gaps:
            return "No additional combined multi-report diagnosis gaps identified."
        return f"Detected {len(combined_gaps)} combined multi-report diagnosis gap(s)."

    def _generate_lab_narrative_summary(
        self,
        llm: Any,
        fallback_llm: Any,
        deduped_report_gaps: list[dict[str, Any]],
        combined_gaps: list[dict[str, Any]],
        problem_list: list[str],
    ) -> str:
        if not deduped_report_gaps and not combined_gaps:
            return "No significant lab abnormalities identified across all reports."
        prompt = PROMPT_LAB_NARRATIVE_SUMMARY.format(
            problem_list_json=json.dumps(problem_list, indent=2),
            lab_gaps_json=json.dumps(deduped_report_gaps, indent=2),
            combined_gaps_json=json.dumps(combined_gaps, indent=2),
        )
        try:
            response = self._invoke_with_fallback(
                primary=llm,
                fallback=fallback_llm,
                fallback_base_llm=fallback_llm,
                messages=[HumanMessage(content=prompt)],
                operation_name="lab narrative summary",
            )
            return getattr(response, "content", str(response)).strip()
        except Exception:
            return self._build_combined_lab_summary(combined_gaps)

    def _generate_doctor_summary(
        self,
        llm: Any,
        fallback_llm: Any,
        medication_gaps: list[dict[str, Any]],
        deduped_report_gaps: list[dict[str, Any]],
        combined_gaps: list[dict[str, Any]],
    ) -> str:
        all_lab_gaps = deduped_report_gaps + combined_gaps
        if not medication_gaps and not all_lab_gaps:
            return "• No gaps identified from medication or lab review."
        prompt = PROMPT_DOCTOR_SUMMARY.format(
            medication_gaps_json=json.dumps(medication_gaps, indent=2),
            lab_gaps_json=json.dumps(all_lab_gaps, indent=2),
        )
        try:
            response = self._invoke_with_fallback(
                primary=llm,
                fallback=fallback_llm,
                fallback_base_llm=fallback_llm,
                messages=[HumanMessage(content=prompt)],
                operation_name="doctor summary",
            )
            return getattr(response, "content", str(response)).strip()
        except Exception:
            lines = [
                f"• Medication: {g.get('medication', '?')} is prescribed but {g.get('implied_condition', '?')} ({g.get('icd10_code', 'UNKNOWN')}) is not documented — confirm diagnosis or review medication appropriateness."
                for g in medication_gaps
            ] + [
                f"• Lab: {g.get('lab_analyte', '?')} {g.get('lab_value', '')} — {g.get('implied_condition', '?')} ({g.get('icd10_code', 'UNKNOWN')})"
                for g in deduped_report_gaps
            ]
            return "\n".join(lines) if lines else "• No coding gaps identified."

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
    def _build_llm(model: str) -> Any:
        if model.startswith("claude-"):
            return ChatAnthropic(model=model, temperature=0)
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
                    f"Primary Groq model '{DEFAULT_HUDDLE_MODEL}' hit a limit during {operation_name}. "
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
                            f"Fallback model '{FALLBACK_HUDDLE_MODEL}' returned non-JSON for {operation_name}. "
                            "Retrying with strict JSON repair."
                        )
                        return self._repair_structured_output_with_base_llm(
                            base_llm=fallback_base_llm,
                            messages=messages,
                            schema_cls=schema_cls,
                            operation_name=operation_name,
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
        # Gemini: langchain-google-genai reads GOOGLE_API_KEY
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key and not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        # Anthropic: langchain-anthropic reads ANTHROPIC_API_KEY (already correct name)

    # ── LLM prompt / response logging ────────────────────────────────────────

    @staticmethod
    def _format_messages_for_log(messages: list[Any]) -> str:
        """Render a list of LangChain messages as readable text."""
        parts: list[str] = []
        for msg in messages:
            role = type(msg).__name__.replace("Message", "").upper()
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                # Multi-part content (tool results etc.)
                content = "\n".join(
                    str(block.get("text", block) if isinstance(block, dict) else block)
                    for block in content
                )
            parts.append(f"[{role}]\n{str(content)}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_response_for_log(response: Any) -> str:
        """Render an LLM response as readable text."""
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
        sep = "-" * 60
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
