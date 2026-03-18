"""Clinical reference guidance injected into per-report and combined lab prompts."""

# This block is always injected as the base into {thresholds_block}.
# For per-report analysis, report-specific reference ranges retrieved via
# DuckDuckGo are appended below this block before the prompt is sent to the LLM.
THRESHOLDS_BLOCK_FALLBACK = """
## Clinical Reference Guidance

Use the report-specific reference ranges provided below as your PRIMARY source for determining abnormal values. Apply your full medical knowledge and clinical reasoning to interpret any values outside their reference ranges and infer undocumented conditions.

General Clinical Interpretation Principles:
- Compare EVERY reported value against its provided reference range
- Flag ANY value outside the normal range — even values just barely outside the range are abnormal
- There is no "too borderline to flag" — if it's outside the range, it's abnormal
- For multi-category reference ranges (e.g., low/intermediate/high risk), flag values not in the optimal/lowest risk category
- For nutritional markers, flag both deficiency AND insufficiency states
- For risk stratification markers, use the risk categories as provided in the reference ranges
- Combine multiple abnormal findings across the same report to infer stronger composite diagnoses where applicable

How to Determine Abnormality:
1. Use the report-specific reference range provided for each lab value (this is your primary authority)
2. If a numeric value falls outside the provided range (lower than minimum OR higher than maximum), it is abnormal
3. For categorical ranges (e.g., "low risk / intermediate risk / high risk"), values outside the best category are abnormal
4. For negative/positive tests, only flag if the result indicates positive/detected status
5. Apply standard clinical interpretation for patterns of abnormalities (e.g., multiple concordant findings suggesting a specific diagnosis)

Instructions:
- Start by examining EVERY value in the report
- Compare each value to its specific reference range provided in the report
- Flag EVERY value that falls outside its reference range
- Do not dismiss borderline abnormalities — abnormal is abnormal
- Combine related abnormal findings to infer specific diagnoses when supported by multiple concordant results
- Use your medical knowledge to interpret what conditions these abnormalities suggest
- Only skip flagging if the value is clearly within the normal/negative range as specified
"""
