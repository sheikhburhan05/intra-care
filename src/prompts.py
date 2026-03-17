"""Prompt templates for patient huddle analysis."""

PROMPT_THRESHOLD_SEARCH = """You have access to a web_search tool. This patient has lab results including: {lab_preview}.

Goal: gather trustworthy diagnostic thresholds/reference ranges for only the labs needed in this case.

Rules:
1. Make 1-3 targeted searches total.
2. Prefer authoritative sources (major guidelines, government/academic/health-system references).
3. Focus on concrete thresholds with units and condition mapping.
4. Avoid generic blogs, forums, or non-clinical content.
5. Prioritize recency when possible.

Use specific queries like:
- "A1C diagnostic criteria diabetes guideline"
- "eGFR CKD staging thresholds guideline"
- "BNP heart failure diagnostic cutoff outpatient"
"""

PROMPT_HUDDLE_ANALYSIS = """You are a clinical decision support assistant. Analyze this patient's medications, lab results, and problem list.
{thresholds_block}

---

## Patient data

{context}

---

Reliability rules (must follow):
1. Use only evidence present in patient data plus provided threshold guidance. Do not invent missing facts.
2. If evidence is weak/ambiguous, do not flag a gap.
3. For medication-based gaps, prefer medications with specific indication. Avoid weak/non-specific mappings.
4. For lab-based gaps, only flag when value clearly crosses a clinically meaningful threshold.
5. If a likely synonym of the condition appears in the problem list, mark in_problem_list=true and do not include it as a suspected gap.
6. Keep evidence short, factual, and threshold-based (include comparator and value when available).
7. Deduplicate repeated findings for the same condition/evidence.
8. This is decision support, not diagnosis.

Perform these analyses:

**1. Medication-to-Diagnosis Mapping**
- Evaluate active medications for high-confidence implied conditions.
- Flag only if the implied condition is not represented in the problem list (including common synonyms).
- Skip low-specificity medications (example: PRN analgesics without disease-specific context).

**2. Lab Value Trigger Engine**
- Parse interpretable lab values and compare against thresholds.
- Only flag if abnormality is clear and condition implication is clinically reasonable.
- Ignore non-diagnostic/qualitative normals (e.g., "NONREACTIVE", "NEGATIVE", "NP") unless the text itself indicates abnormality.
- If unit/context is missing and interpretation is uncertain, skip.

**3. Summary Note Before Huddle**
- Provide a concise pre-huddle summary based only on flagged evidence.
- If no gaps are found, explicitly state that no clear coding gaps were identified from available data.

Output requirements:
- Return valid JSON matching the schema exactly.
- No markdown, no extra keys, no prose outside JSON.
- If no gaps are found, return empty suspected_gaps arrays and clear summaries.
"""

THRESHOLDS_BLOCK_WITH_RESULTS = """
## Retrieved clinical lab thresholds (from web search)
Use these guidelines along with standard clinical knowledge to interpret lab values:

{thresholds_context}
"""

THRESHOLDS_BLOCK_FALLBACK = """
## Lab interpretation
If retrieved thresholds are unavailable, use conservative, commonly accepted clinical reference concepts.
When uncertain, do not over-call a gap.
Consider standard thresholds for: eGFR (CKD), A1C/glucose (diabetes), BNP/NT-proBNP (heart failure),
creatinine (kidney), hemoglobin (anemia), TSH (thyroid), lipids (LDL/HDL), vitamin D, liver enzymes, etc.
"""

PROMPT_SINGLE_REPORT_GAP_ANALYSIS = """You are a clinical coding-gap assistant.
Analyze ONE lab report at a time against the patient's existing problem list.
{thresholds_block}

## Patient Problem List
{problem_list_json}

## Lab Report
Report ID: {report_id}
Results:
{report_results}

Task:
1. Detect clinically meaningful abnormal findings in this report.
2. Infer likely condition(s) suggested by those abnormalities.
3. Compare against the problem list and return only conditions NOT represented in the problem list.

Rules:
- Be conservative. If uncertain, do not flag.
- Use threshold-based evidence when possible (include comparator/value).
- Ignore clearly normal/non-diagnostic qualitative values unless text indicates abnormal.
- Deduplicate repeated findings within this report.
- For every returned condition, include the best matching ICD-10 code in `icd10_code`. If uncertain, use "UNKNOWN".
- Return valid JSON only, matching the schema exactly.
"""

PROMPT_MEDICATION_GAP_ANALYSIS = """You are a clinical coding-gap assistant.
Analyze active medications against the patient's existing problem list.

## Patient Problem List
{problem_list_json}

## Active Medications
{medications_json}

Task:
For each medication, assess whether it is linked to a diagnosis/problem in the current problem list.
Return only medications that need follow-up in one of these categories:
1. Medication strongly implies a condition that is missing from the problem list.
2. Medication appears not clearly linked to any listed diagnosis/problem (possible relevance mismatch).

Rules:
- Be conservative. If uncertain, do not flag.
- Prefer high-specificity medication-condition mappings.
- If a likely synonym exists in the problem list, treat as present (do not flag as missing).
- Evidence must be detailed and explicit. For each flagged medication, include:
  1) the medication name,
  2) the likely indication/condition it is typically used for,
  3) the relevant problem-list comparison (what is present and what is missing),
  4) why this creates a likely diagnosis-linkage gap or relevance mismatch for this patient.
- Do not use vague evidence like "not in list"; explicitly mention the missing or non-matching diagnosis context.
- For every returned condition, include the best matching ICD-10 code in `icd10_code`. If uncertain, use "UNKNOWN".
- Return valid JSON only, matching the schema exactly.
"""

PROMPT_COMBINED_REPORT_GAP_ANALYSIS = """You are a clinical coding-gap assistant focused on multi-report pattern detection.
Your goal: find conditions that emerge ONLY when combining findings from 2+ lab reports—not from any single report alone.
{thresholds_block}

## Patient Problem List
{problem_list_json}

## Per-Report Findings (already reviewed one-by-one)
{per_report_gaps_json}

## Report Snapshots
{all_reports_snapshot}

Task:
1. Look for SYNERGISTIC patterns across reports—where multiple labs together point to a condition that no single report would clearly suggest.
2. Return only conditions NOT represented in the problem list.
3. For each finding, list ALL contributing report IDs and the specific lab values from each that support the pattern.

Examples of multi-report patterns:
- Report 1: elevated fasting glucose; Report 2: elevated HbA1c; Report 3: high triglycerides → diabetes or metabolic syndrome (each alone may be borderline; together they strengthen the picture).
- Report 1: low eGFR; Report 2: elevated creatinine; Report 3: elevated potassium → CKD or acute kidney injury pattern.
- Report 1: low hemoglobin; Report 2: low ferritin; Report 3: low MCV → iron deficiency anemia (combination confirms, not just one value).
- Report 1: elevated BNP; Report 2: elevated creatinine; Report 3: hyponatremia → heart failure with cardiorenal syndrome.

Rules:
- Only flag conditions where the COMBINATION of 2+ reports creates stronger evidence than any single report.
- Be conservative; do not over-call weak or isolated associations.
- In `evidence`, explicitly state which labs from which reports combine to support the condition (e.g., "Report A: fasting glucose 126; Report B: HbA1c 6.8%; Report C: triglycerides 180—together suggest diabetes/metabolic syndrome").
- If condition already exists in problem list (including synonyms), do not return it.
- For every returned condition, include the best matching ICD-10 code in `icd10_code`. If uncertain, use "UNKNOWN".
- Return valid JSON only, matching schema exactly.
"""
