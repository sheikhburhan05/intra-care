"""Orchestrates huddle gap analysis: medication, per-report lab, combined, and doctor summary."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

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
from .llm_client import LLMClient
from .patient_utils import (
    extract_medications,
    extract_problems,
    format_all_reports_for_combined,
    format_report_results,
    short_report_name,
)
from .gap_utils import (
    build_combined_lab_summary,
    build_medication_summary,
    dedupe_combined_lab_gaps,
    dedupe_lab_gaps,
)


# ── Output schemas ─────────────────────────────────────────────────────────────

class SingleReportLabOutput(BaseModel):
    suspected_gaps: list[LabGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Summary of missing diagnosis candidates for this report")


class MedicationAnalysisOutput(BaseModel):
    suspected_gaps: list[MedicationGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Summary of medication-problem list gaps")


class CombinedLabAnalysisOutput(BaseModel):
    suspected_gaps: list[CombinedLabGap] = Field(default_factory=list)
    summary: str = Field(default="", description="Summary of combined multi-report diagnosis gaps")


# ── Main analyzer ──────────────────────────────────────────────────────────────

class HuddleAnalyzer:
    """Run structured huddle analysis for a single patient."""

    def __init__(self, repository: Optional[PatientRepository] = None) -> None:
        self.repository = repository or PatientRepository()
        self.pdf_exporter = HuddlePdfExporter()
        self.llm = LLMClient()

    # ── Public entry point ─────────────────────────────────────────────────────

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
        problem_list = extract_problems(patient)
        medications = extract_medications(patient)
        lab_reports = patient.get("lab_reports", [])

        base_llm = LLMClient.build(model)
        fallback_base_llm = LLMClient.build(FALLBACK_HUDDLE_MODEL)

        def _notify(msg: str) -> None:
            print(msg)
            if progress_callback:
                progress_callback(msg)

        # ── Phase 1 (parallel): medication + all per-report lab analyses ──
        medication_gaps: list[dict[str, Any]] = []
        medication_summary = "Medication analysis skipped."
        deduped_report_gaps: list[dict[str, Any]] = []

        medication_llm = (
            base_llm.with_structured_output(MedicationAnalysisOutput)
            if enable_medication_analysis else None
        )
        fallback_medication_llm = (
            fallback_base_llm.with_structured_output(MedicationAnalysisOutput)
            if enable_medication_analysis else None
        )
        report_llm = (
            base_llm.with_structured_output(SingleReportLabOutput)
            if enable_per_report_lab_analysis else None
        )
        fallback_report_llm = (
            fallback_base_llm.with_structured_output(SingleReportLabOutput)
            if enable_per_report_lab_analysis else None
        )

        valid_reports = (
            [r for r in lab_reports if format_report_results(r)]
            if enable_per_report_lab_analysis else []
        )
        total_reports = len(valid_reports)

        futures: dict[Any, str] = {}
        max_workers = min(1 + len(valid_reports), 12)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            if enable_medication_analysis:
                medication_prompt = PROMPT_MEDICATION_GAP_ANALYSIS.format(
                    problem_list_json=json.dumps(problem_list, indent=2),
                    medications_json=json.dumps(medications, indent=2),
                )
                futures[pool.submit(
                    self.llm.invoke_with_fallback,
                    primary=medication_llm,
                    fallback=fallback_medication_llm,
                    fallback_base_llm=fallback_base_llm,
                    messages=[HumanMessage(content=medication_prompt)],
                    operation_name="medication gap analysis",
                    schema_cls=MedicationAnalysisOutput,
                )] = "__medication__"

            for report in valid_reports:
                report_id = str(report.get("lab_report_id", "")).strip()
                futures[pool.submit(
                    self._analyze_single_report,
                    report=report,
                    report_id=report_id,
                    problem_list=problem_list,
                    report_llm=report_llm,
                    fallback_report_llm=fallback_report_llm,
                    fallback_base_llm=fallback_base_llm,
                    use_web_search=use_web_search,
                )] = report_id or "unknown"

            all_report_gaps: list[dict[str, Any]] = []
            reports_done = 0
            for future in as_completed(futures):
                label = futures[future]
                result = future.result()
                if label == "__medication__":
                    medication_gaps = [gap.model_dump() for gap in result.suspected_gaps]
                    medication_summary = result.summary or build_medication_summary(medication_gaps)
                    _notify(f"✅ Medication Gap Analysis — {len(medication_gaps)} gap(s) found")
                else:
                    reports_done += 1
                    for gap in result.suspected_gaps:
                        gap_dict = gap.model_dump()
                        gap_dict["lab_report_id"] = label or gap_dict.get("lab_report_id", "")
                        all_report_gaps.append(gap_dict)
                    _notify(
                        f"✅ Report {reports_done}/{total_reports}: {short_report_name(label)}"
                        f" — {len(result.suspected_gaps)} gap(s) found"
                    )

        if enable_per_report_lab_analysis:
            deduped_report_gaps = dedupe_lab_gaps(all_report_gaps)

        # ── Phase 2: combined multi-report analysis ────────────────────────────
        combined_gaps: list[dict[str, Any]] = []
        combined_summary = "Combined analysis skipped."
        if enable_combined_lab_analysis or enable_per_report_lab_analysis:
            _notify("⏳ Running combined multi-report analysis…")
            combined_gaps, combined_summary = self._run_combined_lab_analysis(
                base_llm=base_llm,
                fallback_base_llm=fallback_base_llm,
                enable_combined_lab_analysis=enable_combined_lab_analysis,
                lab_reports=lab_reports,
                problem_list=problem_list,
                deduped_report_gaps=deduped_report_gaps,
            )
            _notify(f"✅ Combined analysis — {len(combined_gaps)} pattern(s) found")

        # ── Phase 3: narrative summaries ───────────────────────────────────────
        lab_narrative_summary = ""
        if enable_per_report_lab_analysis or enable_combined_lab_analysis:
            _notify("⏳ Generating lab narrative summary…")
            lab_narrative_summary = self._generate_lab_narrative_summary(
                base_llm=base_llm,
                fallback_base_llm=fallback_base_llm,
                deduped_report_gaps=deduped_report_gaps,
                combined_gaps=combined_gaps,
                problem_list=problem_list,
            )
            _notify("✅ Lab narrative summary complete")

        doctor_summary = ""
        if enable_doctor_summary:
            _notify("⏳ Generating doctor pre-huddle summary…")
            doctor_summary = self._generate_doctor_summary(
                base_llm=base_llm,
                fallback_base_llm=fallback_base_llm,
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
        self.pdf_exporter.export(
            patient_id=pid_key,
            analysis=output,
            output_path=pdf_path,
            enable_medication_analysis=enable_medication_analysis,
            enable_lab_analysis=enable_per_report_lab_analysis or enable_combined_lab_analysis,
            enable_doctor_summary=enable_doctor_summary,
        )
        print(f"Saved huddle PDF to {pdf_path}")
        return output

    # ── Per-report analysis ────────────────────────────────────────────────────

    def _analyze_single_report(
        self,
        report: dict[str, Any],
        report_id: str,
        problem_list: list[str],
        report_llm: Any,
        fallback_report_llm: Any,
        fallback_base_llm: Any,
        use_web_search: bool,
    ) -> Any:
        """Fetch targeted reference ranges for this report, then invoke the LLM.

        Designed to run inside a ThreadPoolExecutor worker thread.
        """
        thresholds_block = THRESHOLDS_BLOCK_FALLBACK
        print(f"\n[DDG] _analyze_single_report: report_id={report_id!r}, use_web_search={use_web_search}")
        if use_web_search and False:
            print(f"[DDG] Web search enabled — calling _fetch_report_thresholds for {report_id!r}")
            report_ranges = self._fetch_report_thresholds(report)
            if report_ranges:
                print(f"[DDG] Appending {len(report_ranges)} chars of report-specific ranges to prompt")
                thresholds_block = (
                    THRESHOLDS_BLOCK_FALLBACK.rstrip()
                    + f"\n\n## Report-Specific Reference Ranges (retrieved via search)\n{report_ranges}"
                )
            else:
                print("[DDG] No report-specific ranges returned — using base thresholds only")
        else:
            print("[DDG] Web search disabled — using base thresholds only")

        report_prompt = PROMPT_SINGLE_REPORT_GAP_ANALYSIS.format(
            thresholds_block=thresholds_block,
            problem_list_json=json.dumps(problem_list, indent=2),
            report_id=report_id or "unknown",
            report_results=format_report_results(report),
        )
        return self.llm.invoke_with_fallback(
            primary=report_llm,
            fallback=fallback_report_llm,
            fallback_base_llm=fallback_base_llm,
            messages=[HumanMessage(content=report_prompt)],
            operation_name=f"per-report lab analysis [{report_id or 'unknown'}]",
            schema_cls=SingleReportLabOutput,
        )

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

            report_name = short_report_name(str(report.get("lab_report_id", "")).strip(), max_len=60)
            analyte_str = ", ".join(analytes[:6])
            query = f"{report_name} {analyte_str} clinical reference range normal values"
            print(f"[DDG] Search query: {query!r}")

            search = DuckDuckGoSearchRun()
            self.llm._log_debug(
                "tool.ddg.per_report_threshold",
                {"input": {"query": query}, "output": {}},
            )
            print("[DDG] Invoking DuckDuckGo search …")
            result = search.invoke(query)
            print(f"[DDG] Search returned {len(result) if result else 0} chars")

            if result and len(result) > 50:
                self.llm._log_debug(
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

    # ── Combined multi-report analysis ─────────────────────────────────────────

    def _run_combined_lab_analysis(
        self,
        base_llm: Any,
        fallback_base_llm: Any,
        enable_combined_lab_analysis: bool,
        lab_reports: list[dict[str, Any]],
        problem_list: list[str],
        deduped_report_gaps: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        if not enable_combined_lab_analysis:
            return [], "Combined multi-report analysis disabled by flag."
        if len(lab_reports) < 2 and len(deduped_report_gaps) < 2:
            return [], "Insufficient multi-report context for combined analysis."

        combined_llm = base_llm.with_structured_output(CombinedLabAnalysisOutput)
        fallback_combined_llm = fallback_base_llm.with_structured_output(CombinedLabAnalysisOutput)

        prompt = PROMPT_COMBINED_REPORT_GAP_ANALYSIS.format(
            problem_list_json=json.dumps(problem_list, indent=2),
            per_report_gaps_json=json.dumps(deduped_report_gaps, indent=2),
            all_reports_snapshot=format_all_reports_for_combined(lab_reports),
        )
        result = self.llm.invoke_with_fallback(
            primary=combined_llm,
            fallback=fallback_combined_llm,
            fallback_base_llm=fallback_base_llm,
            messages=[HumanMessage(content=prompt)],
            operation_name="combined multi-report lab analysis",
            schema_cls=CombinedLabAnalysisOutput,
        )
        combined_gaps = []
        for gap in result.suspected_gaps:
            gap_dict = gap.model_dump()
            gap_dict["contributing_report_ids"] = [
                str(rid).strip()
                for rid in gap_dict.get("contributing_report_ids", [])
                if str(rid).strip()
            ]
            combined_gaps.append(gap_dict)
        deduped = dedupe_combined_lab_gaps(combined_gaps)
        return deduped, (result.summary or build_combined_lab_summary(deduped))

    # ── Narrative summaries ────────────────────────────────────────────────────

    def _generate_lab_narrative_summary(
        self,
        base_llm: Any,
        fallback_base_llm: Any,
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
            response = self.llm.invoke_with_fallback(
                primary=base_llm,
                fallback=fallback_base_llm,
                fallback_base_llm=fallback_base_llm,
                messages=[HumanMessage(content=prompt)],
                operation_name="lab narrative summary",
            )
            return getattr(response, "content", str(response)).strip()
        except Exception:
            return build_combined_lab_summary(combined_gaps)

    def _generate_doctor_summary(
        self,
        base_llm: Any,
        fallback_base_llm: Any,
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
            response = self.llm.invoke_with_fallback(
                primary=base_llm,
                fallback=fallback_base_llm,
                fallback_base_llm=fallback_base_llm,
                messages=[HumanMessage(content=prompt)],
                operation_name="doctor summary",
            )
            return getattr(response, "content", str(response)).strip()
        except Exception:
            lines = [
                f"• Medication: {g.get('medication', '?')} is prescribed but "
                f"{g.get('implied_condition', '?')} ({g.get('icd10_code', 'UNKNOWN')}) is not "
                "documented — confirm diagnosis or review medication appropriateness."
                for g in medication_gaps
            ] + [
                f"• Lab: {g.get('lab_analyte', '?')} {g.get('lab_value', '')} — "
                f"{g.get('implied_condition', '?')} ({g.get('icd10_code', 'UNKNOWN')})"
                for g in deduped_report_gaps
            ]
            return "\n".join(lines) if lines else "• No coding gaps identified."
