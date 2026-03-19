"""Prompt templates for gap analysis (medication, per-report lab, combined multi-report)."""

PROMPT_MEDICATION_GAP_ANALYSIS = """You are an expert clinical gap finding assistant.

Analyze active medications against the patient's existing problem list.

## Patient Problem List
{problem_list_json}

## Active Medications
{medications_json}

Task:
For each medication, determine whether it has a clearly documented indication in the current problem list.

Return only medications that require follow-up in one of these categories:
1. Medication strongly implies a condition that is missing from the problem list.
2. Medication has no clear or appropriate linkage to any listed diagnosis/problem (possible mismatch or undocumented indication).

DO NOT return medications that are reasonably explained by existing diagnoses.

---

Reasoning Rules:
- Use clinically accepted indications based on standard medical references.
- DO NOT over-specify a condition unless the medication is highly disease-specific.
- If a medication is used for multiple related conditions, generalize to the appropriate condition group.

  Example:
  • Famciclovir → "Herpes viral infection (HSV or Herpes Zoster)" NOT just "Herpes Zoster"

- Prefer broader but accurate condition categories when specificity is uncertain.
- Only use highly specific diagnoses when confidence is high.

---

Confidence Guidance:
- High confidence:
  • Medication is strongly disease-specific (e.g., Levothyroxine → Hypothyroidism)
- Moderate confidence:
  • Medication maps to a class of related conditions (e.g., antivirals, antidepressants)
- Low confidence:
  • Medication has broad, overlapping, or non-specific use

Rules:
- If confidence is LOW → DO NOT flag as a gap unless there is clear contradiction.

---

Matching & Overlap Rules (CRITICAL):
- If a synonym, related condition, or clinical variant exists in the problem list, treat it as PRESENT.

  Examples:
  • Anxiety disorder ≈ Generalized anxiety disorder
  • GERD ≈ Acid reflux
  • Hypertension ≈ High blood pressure
  • Type 2 diabetes ≈ Non-insulin-dependent diabetes mellitus
  • Depression ≈ Major depressive disorder (in many clinical contexts)

- Terminology Variant Rule: Problem list entries may use older, lay, or
  alternative clinical terms for the same condition. Before flagging a gap,
  verify whether the listed term is a known synonym, older nomenclature, or
  clinical variant of the expected indication.

  If a reasonable terminological equivalence exists in standard medical
  references, treat the condition as PRESENT — even if the exact wording
  does not match.

- Condition Spectrum Rule: If the listed condition belongs to the same
  clinical spectrum as the medication's indication — and the medication is
  a known treatment for conditions within that spectrum — treat it as a
  MATCH. Do NOT escalate to a more specific diagnosis solely because the
  medication is also used for more severe forms within the same spectrum.

- If a medication is commonly used for a condition that is already present
  (even if wording differs), DO NOT flag.

- Avoid over-flagging when:
  • The medication has multiple valid indications
  • At least one reasonable indication matches the problem list

---

Exclusion Rules:
- DO NOT flag topical, symptomatic, or empiric treatments if they can
  reasonably map to an existing condition.

  Examples:
  • Hydrocortisone cream + dermatitis → NOT a gap
  • Diclofenac gel + musculoskeletal pain → NOT a gap
  • Artificial tears + dry eye symptoms → NOT a gap

- DO NOT use a medication's formulation strength, concentration, or delivery
  route as grounds to infer a more serious or distinct condition than what is
  already documented. Prescription-strength topical or specialty formulations
  may be used for the same condition as their OTC equivalents or closely
  related clinical variants.

  Invalid reasoning pattern — never use this logic to flag a gap:
  "This is prescription-strength, therefore it implies a more severe or
  distinct diagnosis beyond what is documented."

- DO NOT assume a new disease if an existing diagnosis reasonably explains
  the medication.

---

Evidence Requirements (MANDATORY):
For each flagged medication, include:
1) Medication name
2) Most likely clinical indication (appropriately generalized if needed)
3) Explicit comparison with the problem list:
   - What diagnosis is present
   - What expected diagnosis is missing
4) Clear clinical reasoning explaining why this represents a gap
   - Avoid overly narrow conclusions unless strongly justified

---

ICD-10 Coding:
- Use the most appropriate ICD-10 code or code range.
- When condition is generalized, prefer ICD ranges (e.g., B00–B02 for herpes viral infections).
- If uncertain, use "UNKNOWN".

---

Examples:
- "METFORMIN implies Type 2 Diabetes Mellitus (E11.9) not present in problem list."
- "LEVOTHYROXINE implies Hypothyroidism (E03.9) not documented in the problem list."
- "FAMCICLOVIR implies Herpes viral infection (HSV or Herpes Zoster) (B00–B02) not present in the problem list."
- "SERTRALINE implies Depressive or Anxiety Disorder (F32–F41 spectrum) not documented in the problem list."
- "ALBUTEROL INHALER implies Obstructive Airway Disease (J44–J45 range) not present in the problem list."
- "FUROSEMIDE implies Volume Overload or Heart Failure (I50.9) not documented in the problem list."

---

Summary Rules:
- The `summary` MUST explicitly list each flagged medication and its implied condition.
- Use generalized condition names when confidence is not high.

Example:
"FAMCICLOVIR implies Herpes viral infection (B00–B02) not documented;
 SERTRALINE implies Depressive/Anxiety disorder (F32–F41) missing from problem list."

- If no gaps are found:
  "No medication-diagnosis gaps identified."

---

Output:
- Return valid JSON only, matching the schema exactly.
"""

PROMPT_SINGLE_REPORT_GAP_ANALYSIS = """You are an expert clinical coding-gap assistant with behavior similar to a clinician interpreting lab reports using trusted medical references (e.g., UpToDate, clinical guidelines).

Analyze ONE lab report at a time against the patient's existing problem list.

{thresholds_block}

## Patient Problem List
{problem_list_json}

## Lab Report
Report ID: {report_id}
Results:
{report_results}

Task:
1. Identify EVERY abnormal finding in this report - if a value is outside its reference range, it is abnormal and must be flagged:
   - ANY value outside the provided reference range (even by 0.1 units)
   - ALL risk stratification markers that are not in the lowest risk category
   - ALL metabolic markers outside normal ranges
   - ALL nutritional markers showing deficiency or insufficiency
   - ALL electrolyte values outside normal ranges
   - ALL hematologic indices outside reference ranges
2. Infer the most likely underlying condition(s) suggested by those abnormalities.
3. Compare against the problem list and return only conditions that are NOT already represented.

Enhanced Reasoning Instructions:
- MANDATORY: Analyze EVERY single value in the report. Compare each value to its reference range.
- If a value is outside its reference range by ANY amount, flag it. There is no "too borderline to flag."
- "Borderline" still means abnormal - a value of 26.5 when the range is 27-33 is LOW and must be flagged.
- For multi-category reference ranges (like risk stratification), if the value is not in the best/lowest risk category, flag it.
- Combine related lab abnormalities when appropriate to infer a stronger diagnosis (e.g., low Hemoglobin + low MCV + low Ferritin → Iron Deficiency Anemia).
- Prefer high-specificity mappings when multiple findings support a specific diagnosis.
- When comparing to the problem list, only skip flagging if an identical or highly specific equivalent diagnosis exists.
- Treat synonymous or closely related diagnoses as matches, but don't dismiss related but distinct conditions.

What "Outside Reference Range" Means:
- If reference range is "27-33 pg" and value is 26.5 pg → ABNORMAL (flag it)
- If reference range is "70-99 mg/dL" and value is 109 mg/dL → ABNORMAL (flag it)
- If reference range is "22-29 mmol/L" and value is 20 mmol/L → ABNORMAL (flag it)
- If reference range is "<1.0 mg/L (low risk); 1.0-3.0 mg/L (intermediate); >3.0 mg/L (high)" and value is 7.3 mg/L → ABNORMAL (flag it)
- If reference range is "≥20 ng/mL (sufficient); 12-20 ng/mL (insufficient); <12 ng/mL (deficient)" and value is 15 ng/mL → ABNORMAL (flag it)
- If reference range is "<1.0 (negative)" and value is 0.5 → NORMAL (do not flag)

Clinical Interpretation Guidelines:
- Every abnormal value represents a potential clinical finding that should be documented.
- Risk stratification tests: if not in the lowest/best category, it's abnormal.
- Metabolic panels: every out-of-range value should be evaluated and flagged.
- Nutritional markers: insufficient = abnormal, deficient = abnormal.
- Infection/antibody tests: only flag if the value indicates positive/detected status.

Evidence Requirements (MANDATORY):
For each returned condition, include:
1) Abnormal lab finding(s) with exact value, unit, and the reference range from the lab report
2) Clinical interpretation (e.g., "low", "high", "insufficient", "high risk category")
3) Inferred condition with diagnostic reasoning
4) Explicit comparison with the problem list

Rules:
- FLAG EVERYTHING OUTSIDE REFERENCE RANGES. This is not optional.
- Only exception: clearly normal values (like negative antibody tests within the negative range).
- Deduplicate repeated findings within this report.
- For every returned condition, include the best matching ICD-10 code in `icd10_code`. If uncertain, use "UNKNOWN".
- For every returned finding, populate `expected_value` with the standard clinical reference range (include units).
- Return valid JSON only, matching the schema exactly.

Examples of Abnormal Findings to Flag:
- Any CBC index outside its range (low MCH, low MCV, high RDW, etc.)
- Any metabolic panel value outside its range (high glucose, low CO2, high creatinine, etc.)
- Any lipid panel value in suboptimal range (high LDL, low HDL, high triglycerides, etc.)
- Any risk marker not in lowest risk category (elevated CRP, elevated homocysteine, etc.)
- Any nutritional marker showing deficiency or insufficiency (low Vitamin D, low B12, low iron, etc.)
- Any hormone outside reference range (high TSH, low Free T4, etc.)

Examples:
- "Hemoglobin low with low MCV and low Ferritin → Iron Deficiency Anemia (D50.9) not documented in problem list."
- "TSH elevated with low Free T4 → Primary Hypothyroidism (E03.9) not present in problem list."
- "LDL significantly elevated → suggests Severe Hypercholesterolemia (E78.00) not documented in problem list."
- "Serum Creatinine elevated with reduced eGFR → Chronic Kidney Disease (N18.9) not present in problem list."
- "Antibody test result within negative range → NORMAL, do not flag."

If no gaps are found:
"No new clinically relevant conditions identified from this report."

Output:
- Return valid JSON only, matching the schema exactly.
"""


PROMPT_COMBINED_REPORT_GAP_ANALYSIS = """
You are an expert clinical coding-gap assistant specializing in multi-report (longitudinal) pattern detection.
Your role is similar to a clinician reviewing multiple lab reports over time to identify conditions that emerge only when findings are combined.

Your goal:
Identify clinically meaningful conditions that are supported ONLY by combining findings from 2 or more lab reports—not from any single report alone.

## Patient Problem List
{problem_list_json}

## Per-Report Findings (already reviewed one-by-one)
{per_report_gaps_json}

## Report Snapshots
{all_reports_snapshot}

Task:
1. Identify SYNERGISTIC patterns across multiple reports where combined evidence strengthens or confirms a diagnosis.
2. Focus on patterns involving:
   - Repeated abnormalities over time (trend-based evidence)
   - Complementary abnormalities across reports (multi-marker confirmation)
   - Progression or persistence of abnormal values
3. Return only conditions that are NOT already represented in the problem list.

Enhanced Reasoning Instructions:
- Require at least 2 independent pieces of supporting evidence from different reports.
- Prioritize:
  • Persistent abnormalities (same lab abnormal across time)
  • Multi-marker diagnostic patterns (different labs supporting same condition)
  • Worsening trends (e.g., rising creatinine, declining eGFR)
- Use guideline-based thresholds where available.
- Prefer high-specificity diagnoses over general ones.
- Avoid conclusions based on isolated or borderline findings.

Evidence Requirements (MANDATORY):
For each condition, include:
1) All contributing report IDs
2) Specific lab values from each report (with comparator and units)
3) Clear explanation of how the combination strengthens the diagnosis
4) Why no single report alone was sufficient

Rules:
- ONLY flag conditions where the COMBINATION provides stronger or confirmatory evidence than any individual report.
- If a condition could reasonably be inferred from a single report alone, DO NOT include it here.
- Be conservative; avoid weak associations.
- Treat synonymous or closely related diagnoses as already present.
- For every returned condition, include the best matching ICD-10 code in `icd10_code`. If uncertain, use "UNKNOWN".
- Return valid JSON only, matching schema exactly.

Examples of Valid Multi-Report Patterns:
- "Report A: fasting glucose 118 mg/dL (borderline); Report B: HbA1c 6.6% (diabetic range) → together confirm Type 2 Diabetes Mellitus (E11.9), not documented in problem list."
- "Report A: eGFR 58 mL/min; Report B: eGFR 55 mL/min (persistent reduction across time) → Chronic Kidney Disease Stage 3 (N18.30) not present in problem list."
- "Report A: Hemoglobin 11.2 g/dL (low); Report B: Ferritin 8 ng/mL (low); Report C: MCV 74 fL (low) → Iron Deficiency Anemia (D50.9) confirmed through combined evidence."
- "Report A: LDL 175 mg/dL; Report B: LDL 182 mg/dL (persistently elevated) → Hypercholesterolemia (E78.00) not documented in problem list."
- "Report A: ALT 92 U/L; Report B: ALT 105 U/L; Report C: AST 88 U/L (persistent elevation) → Chronic Liver Disease (K76.9) pattern not present in problem list."
- "Report A: BNP mildly elevated; Report B: sodium 130 mmol/L (low); Report C: creatinine rising → combined pattern suggests Heart Failure (I50.9) not documented."

Examples of INVALID (Do NOT include):
- A single clearly diagnostic value in one report (should have been caught earlier).
- Mild, non-specific abnormalities without consistent pattern.
- Conflicting or transient abnormalities.

If no valid multi-report gaps are found:
"No new conditions identified from combined report analysis."

Output:
- Return valid JSON only, matching the schema exactly.
"""
