"""Service for medication/lab-to-diagnosis gap detection via LLM."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..config import DEFAULT_HUDDLE_MODEL, resolve_repo_path
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
        llm = ChatAnthropic(model=model, temperature=0)
        report_llm = llm.with_structured_output(SingleReportLabOutput)
        medication_llm = llm.with_structured_output(MedicationAnalysisOutput)
        combined_llm = llm.with_structured_output(CombinedLabAnalysisOutput)

        medication_prompt = PROMPT_MEDICATION_GAP_ANALYSIS.format(
            problem_list_json=json.dumps(problem_list, indent=2),
            medications_json=json.dumps(medications, indent=2),
        )
        medication_result = medication_llm.invoke([HumanMessage(content=medication_prompt)])
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
            report_result = report_llm.invoke([HumanMessage(content=report_prompt)])
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
        result = combined_llm.invoke([HumanMessage(content=prompt)])
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
            return self._fetch_clinical_thresholds_with_llm_tools(labs, model)
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
            llm = ChatAnthropic(model=model, temperature=0)
            llm_with_tools = llm.bind_tools([search_tool])

            lab_names = list(dict.fromkeys(lab.get("labanalyte", "") for lab in labs if lab.get("labanalyte")))[:30]
            lab_preview = ", ".join(lab_names) if lab_names else "various lab tests"
            prompt = PROMPT_THRESHOLD_SEARCH.format(lab_preview=lab_preview)

            messages = [HumanMessage(content=prompt)]
            all_results: list[str] = []
            for _ in range(5):
                response = llm_with_tools.invoke(messages)
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
                    except Exception:
                        output = "[Search failed]"
                        all_results.append(f"Query: {query}\n[Search failed]")
                    tool_messages.append(ToolMessage(content=output, tool_call_id=tool_call.get("id", str(index))))
                messages.append(response)
                messages.extend(tool_messages)

            if all_results:
                return "\n\n---\n\n".join(all_results) + "\n\nUse the above retrieved guidelines where applicable."
            return ""
        except ImportError:
            return ""
        except Exception:
            return ""

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
