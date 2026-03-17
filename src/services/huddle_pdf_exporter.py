"""Generate a clinician-friendly PDF summary from huddle analysis output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

LineType = Literal["title", "heading", "subheading", "body", "blank", "separator"]


class HuddlePdfExporter:
    """Export huddle analysis into a readable PDF with clear headings and spacing."""

    # Layout constants (points)
    PAGE_WIDTH = 612
    PAGE_HEIGHT = 792
    MARGIN_LEFT = 50
    MARGIN_RIGHT = 50
    MARGIN_TOP = 720
    MARGIN_BOTTOM = 50
    LINE_WIDTH = 95  # chars for wrapping

    # Font sizes
    TITLE_SIZE = 18
    HEADING_SIZE = 14
    SUBHEADING_SIZE = 12
    BODY_SIZE = 10

    # Line heights (points)
    TITLE_LEADING = 24
    HEADING_LEADING = 18
    SUBHEADING_LEADING = 16
    BODY_LEADING = 13
    BLANK_LEADING = 8
    SEPARATOR_LEADING = 12

    def export(self, patient_id: str, analysis: dict[str, Any], output_path: str | Path) -> Path:
        path = Path(output_path)
        blocks = self._build_content_blocks(patient_id, analysis)
        pdf_bytes = self._build_pdf(blocks)
        path.write_bytes(pdf_bytes)
        return path

    def _build_content_blocks(self, patient_id: str, analysis: dict[str, Any]) -> list[tuple[LineType, str]]:
        """Return list of (line_type, text) for PDF rendering."""
        medication = analysis.get("medication_to_diagnosis", {})
        lab = analysis.get("lab_report_to_diagnosis", {})
        combined = analysis.get("combined_lab_report_to_diagnosis", {})
        note = analysis.get("summary_note_before_huddle", {})

        blocks: list[tuple[LineType, str]] = [
            ("title", f"Patient Huddle Summary"),
            ("subheading", f"Patient ID: {patient_id}"),
            ("blank", ""),
            ("separator", "-" * 40),
            ("blank", ""),
            ("heading", "1. Medication to Diagnosis Gaps"),
            ("body", medication.get("summary", "N/A")),
            ("blank", ""),
        ]

        med_gaps = medication.get("suspected_gaps", []) or []
        if not med_gaps:
            blocks.append(("body", "No medication-diagnosis gaps identified."))
        else:
            for i, gap in enumerate(med_gaps, start=1):
                blocks.extend([
                    ("subheading", f"{i}. {gap.get('medication', 'N/A')}"),
                    ("body", f"Implied condition: {gap.get('implied_condition', 'N/A')}"),
                    ("body", f"ICD-10: {gap.get('icd10_code', 'UNKNOWN')}"),
                    ("body", f"Evidence: {gap.get('evidence', 'N/A')}"),
                    ("blank", ""),
                ])

        blocks.extend([
            ("separator", "-" * 40),
            ("blank", ""),
            ("heading", "2. Per-Report Lab to Diagnosis Gaps"),
            ("body", lab.get("summary", "N/A")),
            ("blank", ""),
        ])

        lab_gaps = lab.get("suspected_gaps", []) or []
        if not lab_gaps:
            blocks.append(("body", "No per-report lab gaps identified."))
        else:
            for i, gap in enumerate(lab_gaps, start=1):
                blocks.extend([
                    ("subheading", f"{i}. {gap.get('lab_analyte', 'N/A')} = {gap.get('lab_value', 'N/A')}"),
                    ("body", f"Report: {gap.get('lab_report_id', 'N/A')}"),
                    ("body", f"Implied condition: {gap.get('implied_condition', 'N/A')}"),
                    ("body", f"ICD-10: {gap.get('icd10_code', 'UNKNOWN')}"),
                    ("body", f"Evidence: {gap.get('evidence', 'N/A')}"),
                    ("blank", ""),
                ])

        blocks.extend([
            ("separator", "-" * 40),
            ("blank", ""),
            ("heading", "3. Combined Multi-Report Signals"),
            ("body", combined.get("summary", "N/A")),
            ("blank", ""),
        ])

        combined_gaps = combined.get("suspected_gaps", []) or []
        if not combined_gaps:
            blocks.append(("body", "No combined multi-report gaps identified."))
        else:
            for i, gap in enumerate(combined_gaps, start=1):
                report_ids = ", ".join(gap.get("contributing_report_ids", []) or [])
                blocks.extend([
                    ("subheading", f"{i}. {gap.get('implied_condition', 'N/A')}"),
                    ("body", f"ICD-10: {gap.get('icd10_code', 'UNKNOWN')}"),
                    ("body", f"Contributing reports: {report_ids or 'N/A'}"),
                    ("body", f"Evidence: {gap.get('evidence', 'N/A')}"),
                    ("blank", ""),
                ])

        blocks.extend([
            ("separator", "-" * 40),
            ("blank", ""),
            ("heading", "4. Suggested Huddle Note"),
            ("subheading", "Context"),
            ("body", note.get("context", "N/A")),
            ("blank", ""),
            ("subheading", "Suggested bullet"),
            ("body", note.get("suggested_huddle_note_bullet", "N/A")),
            ("blank", ""),
            ("subheading", "Physician prompt"),
            ("body", note.get("physician_prompt", "N/A")),
        ])

        return blocks

    def _build_pdf(self, blocks: list[tuple[LineType, str]]) -> bytes:
        """Build PDF with proper headings, spacing, and fonts."""
        # Expand blocks into drawable lines with font/size info
        draw_items: list[tuple[str, int, bool, float]] = []  # (text, font_size, bold, leading)
        for line_type, text in blocks:
            if line_type == "blank":
                draw_items.append(("", self.BODY_SIZE, False, self.BLANK_LEADING))
                continue
            if line_type == "separator":
                draw_items.append((text[: self.LINE_WIDTH], self.BODY_SIZE, False, self.SEPARATOR_LEADING))
                continue

            wrapped = self._wrap_line(text, self.LINE_WIDTH)
            if line_type == "title":
                for w in wrapped:
                    draw_items.append((w, self.TITLE_SIZE, True, self.TITLE_LEADING))
            elif line_type == "heading":
                for w in wrapped:
                    draw_items.append((w, self.HEADING_SIZE, True, self.HEADING_LEADING))
            elif line_type == "subheading":
                for w in wrapped:
                    draw_items.append((w, self.SUBHEADING_SIZE, True, self.SUBHEADING_LEADING))
            else:
                for w in wrapped:
                    draw_items.append((w, self.BODY_SIZE, False, self.BODY_LEADING))

        # Paginate
        max_height = self.MARGIN_TOP - self.MARGIN_BOTTOM
        pages_content: list[list[tuple[str, int, bool, float]]] = []
        current_page: list[tuple[str, int, bool, float]] = []
        current_y = 0

        for item in draw_items:
            _, _, _, leading = item
            if current_y + leading > max_height and current_page:
                pages_content.append(current_page)
                current_page = []
                current_y = 0
            current_page.append(item)
            current_y += leading

        if current_page:
            pages_content.append(current_page)

        # PDF objects: 1=catalog, 2=pages, 3=font regular, 4=font bold
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        kids_refs = " ".join(f"{5 + i * 2} 0 R" for i in range(len(pages_content)))
        objects.append(f"<< /Type /Pages /Count {len(pages_content)} /Kids [{kids_refs}] >>".encode("ascii"))
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

        for i, page_items in enumerate(pages_content):
            page_obj_num = 5 + i * 2
            content_obj_num = 6 + i * 2
            objects.append(
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.PAGE_WIDTH} {self.PAGE_HEIGHT}] "
                    f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_obj_num} 0 R >>"
                ).encode("ascii")
            )

            stream_lines = ["BT"]
            y = self.MARGIN_TOP
            for text, font_size, bold, leading in page_items:
                font = "/F2" if bold else "/F1"
                stream_lines.append(f"{font} {font_size} Tf")
                safe = self._to_pdf_safe(text)
                escaped = self._pdf_escape(safe)
                stream_lines.append(f"1 0 0 1 {self.MARGIN_LEFT} {y} Tm ({escaped}) Tj")
                y -= leading
            stream_lines.append("ET")

            stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
            objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream")

        return self._assemble_pdf(objects)

    @staticmethod
    def _to_pdf_safe(text: str) -> str:
        """Convert Unicode to Latin-1-safe chars (Helvetica supports Latin-1 only)."""
        replacements = [
            ("\u2014", "-"),   # em dash
            ("\u2013", "-"),   # en dash
            ("\u2018", "'"),   # left single quote
            ("\u2019", "'"),   # right single quote
            ("\u201c", '"'),   # left double quote
            ("\u201d", '"'),   # right double quote
            ("\u2026", "..."), # ellipsis
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    @staticmethod
    def _pdf_escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    @staticmethod
    def _wrap_line(text: str, width: int) -> list[str]:
        if len(text) <= width:
            return [text] if text else [""]
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= width:
                current = candidate
                continue
            lines.append(current)
            current = word
        lines.append(current)
        return lines

    @staticmethod
    def _assemble_pdf(objects: list[bytes]) -> bytes:
        parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
        offsets = [0]
        current = len(parts[0])
        for idx, obj in enumerate(objects, start=1):
            entry = f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
            offsets.append(current)
            parts.append(entry)
            current += len(entry)

        xref_start = current
        xref = [f"xref\n0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
        for off in offsets[1:]:
            xref.append(f"{off:010d} 00000 n \n".encode("ascii"))
        parts.extend(xref)
        trailer = (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii")
        )
        parts.append(trailer)
        return b"".join(parts)
