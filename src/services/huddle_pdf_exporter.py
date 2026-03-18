"""Generate a clinician-friendly PDF summary from huddle analysis output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

LineType = Literal["title", "heading", "subheading", "body", "body_kv", "blank", "separator"]


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
        doctor_summary = analysis.get("doctor_summary", "")

        blocks: list[tuple[LineType, str]] = [
            ("title", "Patient Huddle Summary"),
            ("subheading", f"Patient ID: {patient_id}"),
            ("blank", ""),
            ("separator", "-" * 40),
            ("blank", ""),
        ]

        # ── Section 1: Medication to Diagnosis ──────────────────────────────
        blocks.extend([
            ("heading", "Medication to Diagnosis:"),
            ("blank", ""),
        ])

        med_gaps = medication.get("suspected_gaps", []) or []
        med_summary = medication.get("summary", "").strip()
        if med_summary:
            blocks.extend([
                ("subheading", "Summary of findings:"),
                ("body", med_summary),
                ("blank", ""),
            ])

        if not med_gaps:
            blocks.append(("body", "No medication-diagnosis gaps identified."))
        else:
            for i, gap in enumerate(med_gaps, start=1):
                blocks.extend([
                    ("subheading", f"Finding {i}:"),
                    ("body_kv", "Medication:", gap.get("medication", "N/A")),
                    ("body_kv", "Missing diagnosis:", f"{gap.get('implied_condition', 'N/A')} ({gap.get('icd10_code', 'UNKNOWN')})"),
                    ("body_kv", "Evidence:", gap.get("evidence", "N/A")),
                    ("blank", ""),
                ])

        # ── Section 2: Lab Report Analysis ──────────────────────────────────
        blocks.extend([
            ("separator", "-" * 40),
            ("blank", ""),
            ("heading", "Lab Report Analysis:"),
            ("blank", ""),
        ])

        lab_gaps = lab.get("suspected_gaps", []) or []
        if not lab_gaps:
            blocks.append(("body", "No abnormal lab findings identified."))
        else:
            for i, gap in enumerate(lab_gaps, start=1):
                expected = gap.get("expected_value", "").strip() or "N/A"
                blocks.extend([
                    ("subheading", f"Abnormal lab result {i}:"),
                    ("body_kv", "Lab Test Name:", gap.get("lab_report_id", "N/A")),
                    ("body_kv", "Labanalyte:", gap.get("lab_analyte", "N/A")),
                    ("body_kv", "Lab Value:", gap.get("lab_value", "N/A")),
                    ("body_kv", "Expected Value:", expected),
                    ("blank", ""),
                ])

        narrative = lab.get("narrative_summary", "").strip()
        if narrative:
            blocks.extend([
                ("subheading", "Summary of Lab Report findings:"),
                ("body", narrative),
                ("blank", ""),
            ])

        # ── Section 3: Doctor Summary ────────────────────────────────────────
        blocks.extend([
            ("separator", "-" * 40),
            ("blank", ""),
            ("heading", "Pre-Huddle Summary:"),
            ("blank", ""),
        ])
        if doctor_summary:
            for line in doctor_summary.splitlines():
                line = line.strip()
                if line:
                    blocks.append(("body", line))
                    blocks.append(("blank", ""))
        else:
            blocks.append(("body", "No gaps identified."))

        return blocks

    def _build_pdf(self, blocks: list[tuple]) -> bytes:
        """Build PDF with proper headings, spacing, and fonts."""
        # Each draw item: (segments, font_size, leading)
        # segments: list of (text, is_bold) — allows mixed bold/regular on one line
        draw_items: list[tuple[list[tuple[str, bool]], int, float]] = []

        for block in blocks:
            line_type = block[0]

            if line_type == "blank":
                draw_items.append(([], self.BODY_SIZE, self.BLANK_LEADING))
                continue
            if line_type == "separator":
                text = block[1]
                draw_items.append(([(text[: self.LINE_WIDTH], False)], self.BODY_SIZE, self.SEPARATOR_LEADING))
                continue

            if line_type == "body_kv":
                # block = ("body_kv", label, value)
                label = str(block[1])
                value = str(block[2]) if len(block) > 2 else ""
                kv_text = f"{label}  {value}"
                wrapped = self._wrap_kv_line(label, value, self.LINE_WIDTH)
                for idx, (lbl_seg, val_seg) in enumerate(wrapped):
                    segs: list[tuple[str, bool]] = []
                    if lbl_seg:
                        segs.append((lbl_seg, True))
                    if val_seg:
                        segs.append((val_seg, False))
                    if not segs:
                        segs = [(kv_text, False)]
                    draw_items.append((segs, self.BODY_SIZE, self.BODY_LEADING))
                continue

            text = block[1]
            if line_type == "title":
                for w in self._wrap_line(text, self.LINE_WIDTH):
                    draw_items.append(([(w, True)], self.TITLE_SIZE, self.TITLE_LEADING))
            elif line_type == "heading":
                for w in self._wrap_line(text, self.LINE_WIDTH):
                    draw_items.append(([(w, True)], self.HEADING_SIZE, self.HEADING_LEADING))
            elif line_type == "subheading":
                for w in self._wrap_line(text, self.LINE_WIDTH):
                    draw_items.append(([(w, True)], self.SUBHEADING_SIZE, self.SUBHEADING_LEADING))
            else:
                # Parse **...** inline bold markers for body text
                inline_segs = self._parse_inline_bold(text)
                for line_segs in self._wrap_rich_segments(inline_segs, self.LINE_WIDTH):
                    draw_items.append((line_segs, self.BODY_SIZE, self.BODY_LEADING))

        # Paginate
        max_height = self.MARGIN_TOP - self.MARGIN_BOTTOM
        pages_content: list[list[tuple[list[tuple[str, bool]], int, float]]] = []
        current_page: list[tuple[list[tuple[str, bool]], int, float]] = []
        current_y = 0.0

        for item in draw_items:
            _, _, leading = item
            if current_y + leading > max_height and current_page:
                pages_content.append(current_page)
                current_page = []
                current_y = 0.0
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
            content_obj_num = 6 + i * 2
            objects.append(
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.PAGE_WIDTH} {self.PAGE_HEIGHT}] "
                    f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_obj_num} 0 R >>"
                ).encode("ascii")
            )

            stream_lines = ["BT"]
            y = self.MARGIN_TOP
            for segments, font_size, leading in page_items:
                if not segments:
                    y -= leading
                    continue
                # Position cursor at start of this line
                first_font = "/F2" if segments[0][1] else "/F1"
                stream_lines.append(f"{first_font} {font_size} Tf")
                stream_lines.append(f"1 0 0 1 {self.MARGIN_LEFT} {y} Tm")
                for seg_text, seg_bold in segments:
                    font = "/F2" if seg_bold else "/F1"
                    stream_lines.append(f"{font} {font_size} Tf")
                    safe = self._to_pdf_safe(seg_text)
                    escaped = self._pdf_escape(safe)
                    stream_lines.append(f"({escaped}) Tj")
                y -= leading
            stream_lines.append("ET")

            stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
            objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream")

        return self._assemble_pdf(objects)

    @staticmethod
    def _to_pdf_safe(text: str) -> str:
        """Convert Unicode to Latin-1-safe chars (Helvetica supports Latin-1 only)."""
        replacements = [
            ("\u2022", "-"),   # bullet point
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
    def _parse_inline_bold(text: str) -> list[tuple[str, bool]]:
        """Split text on **...** markers into (text, is_bold) segments."""
        parts = text.split("**")
        return [(part, i % 2 == 1) for i, part in enumerate(parts) if part]

    @staticmethod
    def _wrap_rich_segments(
        segments: list[tuple[str, bool]], width: int
    ) -> list[list[tuple[str, bool]]]:
        """Word-wrap a list of (text, bold) segments to lines of at most `width` chars.

        Consecutive same-bold words on the same line are merged into one segment.
        """
        # Flatten to a word list preserving bold flag
        words: list[tuple[str, bool]] = []
        for text, bold in segments:
            for word in text.split():
                words.append((word, bold))

        if not words:
            return [[("", False)]]

        lines: list[list[tuple[str, bool]]] = []
        current_words: list[tuple[str, bool]] = []
        current_len = 0

        for word, bold in words:
            needed = current_len + (1 if current_words else 0) + len(word)
            if current_words and needed > width:
                lines.append(HuddlePdfExporter._merge_word_segments(current_words))
                current_words = [(word, bold)]
                current_len = len(word)
            else:
                current_words.append((word, bold))
                current_len = needed

        if current_words:
            lines.append(HuddlePdfExporter._merge_word_segments(current_words))
        return lines

    @staticmethod
    def _merge_word_segments(words: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        """Join consecutive same-bold words with spaces into single segments."""
        if not words:
            return []
        result: list[tuple[str, bool]] = []
        cur_text, cur_bold = words[0]
        for word, bold in words[1:]:
            if bold == cur_bold:
                cur_text += " " + word
            else:
                result.append((cur_text, cur_bold))
                cur_text, cur_bold = word, bold
        result.append((cur_text, cur_bold))
        return result

    @staticmethod
    def _wrap_kv_line(label: str, value: str, width: int) -> list[tuple[str, str]]:
        """Wrap a key-value pair; returns list of (label_part, value_part) segment pairs.

        The label (bold) appears only on the first wrapped line. Continuation lines
        are indented to align with the value start.
        """
        label_with_space = label + "  "
        indent = " " * len(label_with_space)
        first_value_width = width - len(label_with_space)
        if first_value_width < 10:
            first_value_width = width

        words = value.split()
        if not words:
            return [(label_with_space, "")]

        lines: list[tuple[str, str]] = []
        current = words[0]
        is_first = True
        for word in words[1:]:
            candidate = f"{current} {word}"
            limit = first_value_width if is_first else (width - len(indent))
            if len(candidate) <= limit:
                current = candidate
            else:
                lines.append((label_with_space if is_first else indent, current))
                is_first = False
                current = word
        lines.append((label_with_space if is_first else indent, current))
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
