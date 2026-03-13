"""Service for medication/lab-to-diagnosis gap detection via LLM."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from ..config import DEFAULT_HUDDLE_MODEL, resolve_repo_path
from ..domain.huddle_output import HuddleAnalysisOutput
from ..prompts import (
    PROMPT_HUDDLE_ANALYSIS,
    PROMPT_THRESHOLD_SEARCH,
    THRESHOLDS_BLOCK_FALLBACK,
    THRESHOLDS_BLOCK_WITH_RESULTS,
)
from ..repositories.patient_repository import PatientRepository


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
        context = self._build_patient_context(patient)
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
        prompt = PROMPT_HUDDLE_ANALYSIS.format(thresholds_block=thresholds_block, context=context)

        llm = ChatAnthropic(model=model, temperature=0)
        structured_llm = llm.with_structured_output(HuddleAnalysisOutput)
        result = structured_llm.invoke([HumanMessage(content=prompt)])

        output = {
            "patient_id": patient_id,
            "medication_to_diagnosis": {
                "suspected_gaps": [gap.model_dump() for gap in result.medication_to_diagnosis.suspected_gaps],
                "summary": result.medication_to_diagnosis.summary,
            },
            "lab_report_to_diagnosis": {
                "suspected_gaps": [gap.model_dump() for gap in result.lab_report_to_diagnosis.suspected_gaps],
                "summary": result.lab_report_to_diagnosis.summary,
            },
            "summary_note_before_huddle": result.summary_note_before_huddle.model_dump(),
        }

        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / f"{pid_key}.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)
        print(f"Saved huddle analysis to {out_path}")
        return output

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

    def _build_patient_context(self, patient: dict[str, Any]) -> str:
        medications = self._extract_medications(patient)
        problems = self._extract_problems(patient)
        labs = self._extract_labs(patient)
        lines = [
            "## Patient Problem List (ICD-10 / SNOMED descriptions)",
            json.dumps(problems, indent=2),
            "",
            "## Active Medications",
            json.dumps(medications, indent=2),
            "",
            "## Lab Results (analyte: value)",
        ]
        for lab in labs[:80]:
            lines.append(f"  - {lab['labanalyte']}: {lab['labvalue']}")
        if len(labs) > 80:
            lines.append(f"  ... and {len(labs) - 80} more")
        return "\n".join(lines)
