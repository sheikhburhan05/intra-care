"""Prompt templates for patient huddle analysis."""

PROMPT_THRESHOLD_SEARCH = """You have access to a web_search tool. This patient has lab results including: {lab_preview}.

Use the web_search tool to look up current clinical reference ranges and diagnostic thresholds for the lab tests you'll need to interpret. Search for guidelines on eGFR, A1C, BNP, creatinine, hemoglobin, TSH, lipids, etc. as relevant.

Call the tool with specific search queries (e.g. "eGFR CKD threshold 2024", "A1C diabetes diagnostic criteria"). Make 1-3 searches to gather threshold information."""

PROMPT_HUDDLE_ANALYSIS = """You are a clinical decision support assistant. Analyze this patient's medications, lab results, and problem list.
{thresholds_block}

---

## Patient data

{context}

---

Perform two analyses:

**1. Medication-to-Diagnosis Mapping**
- For each medication, determine what condition(s) it typically treats (e.g., insulin -> diabetes, metformin -> diabetes, levothyroxine -> hypothyroidism).
- If a medication implies a condition that is NOT documented in the patient's problem list, flag it as a suspected gap.
- For each gap: state the medication, implied condition, evidence, and whether that condition IS or IS NOT in the patient's problem list.

**2. Lab Value Trigger Engine**
- For lab results with numeric or interpretable values, apply clinical thresholds to identify potential conditions (e.g., eGFR < 60 -> CKD, A1C elevated -> diabetes).
- If an abnormal lab suggests a condition NOT in the problem list, flag it.
- For each gap: state the lab analyte, value, implied condition, evidence (threshold), and whether that condition IS or IS NOT in the problem list.
- Ignore labs that are clearly normal or non-diagnostic (e.g., "NONREACTIVE", "NP", normal CBC ranges).

**3. Summary Note Before Huddle**
- Synthesize the flags into: context supporting the flags, a suggested huddle note bullet, and a short physician-facing prompt/coding reminder.

Output valid JSON matching the schema. If no gaps are found, return empty lists with appropriate summaries."""

THRESHOLDS_BLOCK_WITH_RESULTS = """
## Retrieved clinical lab thresholds (from web search)
Use these guidelines along with standard clinical knowledge to interpret lab values:

{thresholds_context}
"""

THRESHOLDS_BLOCK_FALLBACK = """
## Lab interpretation
Apply your knowledge of current clinical guidelines and reference ranges to interpret lab values.
Consider standard thresholds for: eGFR (CKD), A1C/glucose (diabetes), BNP/NT-proBNP (heart failure),
creatinine (kidney), hemoglobin (anemia), TSH (thyroid), lipids (LDL/HDL), vitamin D, liver enzymes, etc.
"""
