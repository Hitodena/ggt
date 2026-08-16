"""Extract plain text from uploaded office/PDF files."""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import pymupdf
import openpyxl
import pytesseract
import xlrd
from docx import Document
from PIL import Image

from app.core.config import Settings, get_settings

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".doc", ".docx", ".xls", ".xlsx"})


class ExtractionError(RuntimeError):
    pass


class TextExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if self.settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_cmd

    def extract(self, filename: str, data: bytes) -> list[str]:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ExtractionError(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if not data:
            raise ExtractionError("Empty file")

        if ext == ".pdf":
            return self._extract_pdf(data)
        if ext == ".docx":
            return self._extract_docx(data)
        if ext == ".doc":
            return self._extract_doc(data)
        if ext == ".xlsx":
            return self._extract_xlsx(data)
        if ext == ".xls":
            return self._extract_xls(data)
        raise ExtractionError(f"Unsupported file type '{ext}'")

    def _extract_pdf(self, data: bytes) -> list[str]:
        blocks: list[str] = []
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"Failed to open PDF: {exc}") from exc

        with doc:
            for page_index, page in enumerate(doc, start=1):
                text = (page.get_text("text") or "").strip()
                if len(text) < self.settings.pdf_ocr_min_chars:
                    text = self._ocr_pdf_page(page)
                text = text.strip()
                if text:
                    blocks.append(f"[Страница {page_index}]\n{text}")

        if not blocks:
            raise ExtractionError("No text could be extracted from PDF")
        return blocks

    def _ocr_pdf_page(self, page: pymupdf.Page) -> str:
        try:
            pixmap = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            return pytesseract.image_to_string(
                image,
                lang=self.settings.ocr_languages,
            )
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"OCR failed: {exc}") from exc

    def _extract_docx(self, data: bytes) -> list[str]:
        try:
            document = Document(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"Failed to open DOCX: {exc}") from exc

        parts: list[str] = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)

        for table_index, table in enumerate(document.tables, start=1):
            rows: list[str] = []
            for row in table.rows:
                cells = [
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                ]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                parts.append(f"[Таблица {table_index}]\n" + "\n".join(rows))

        if not parts:
            raise ExtractionError("No text could be extracted from DOCX")
        return parts

    def _extract_doc(self, data: bytes) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "input.doc"
            source.write_bytes(data)
            cmd = [
                self.settings.libreoffice_path,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(tmp_path),
                str(source),
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.settings.libreoffice_timeout_sec,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise ExtractionError(
                    "LibreOffice (soffice) not found. "
                    "Install LibreOffice or set LIBREOFFICE_PATH."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise ExtractionError(
                    "LibreOffice conversion timed out"
                ) from exc

            converted = tmp_path / "input.docx"
            if completed.returncode != 0 or not converted.exists():
                stderr = (completed.stderr or completed.stdout or "").strip()
                raise ExtractionError(
                    "Failed to convert .doc via LibreOffice"
                    + (f": {stderr}" if stderr else "")
                )
            return self._extract_docx(converted.read_bytes())

    def _extract_xlsx(self, data: bytes) -> list[str]:
        try:
            workbook = openpyxl.load_workbook(
                io.BytesIO(data),
                read_only=True,
                data_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"Failed to open XLSX: {exc}") from exc

        blocks: list[str] = []
        try:
            for sheet in workbook.worksheets:
                lines: list[str] = []
                for row in sheet.iter_rows(values_only=True):
                    cells = [
                        str(cell).strip()
                        for cell in row
                        if cell is not None and str(cell).strip()
                    ]
                    if cells:
                        lines.append(" | ".join(cells))
                if lines:
                    blocks.append(f"[Лист {sheet.title}]\n" + "\n".join(lines))
        finally:
            workbook.close()

        if not blocks:
            raise ExtractionError("No text could be extracted from XLSX")
        return blocks

    def _extract_xls(self, data: bytes) -> list[str]:
        try:
            book = xlrd.open_workbook(file_contents=data)
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"Failed to open XLS: {exc}") from exc

        blocks: list[str] = []
        for sheet in book.sheets():
            lines: list[str] = []
            for row_index in range(sheet.nrows):
                cells = [
                    str(sheet.cell_value(row_index, col_index)).strip()
                    for col_index in range(sheet.ncols)
                    if str(sheet.cell_value(row_index, col_index)).strip()
                ]
                if cells:
                    lines.append(" | ".join(cells))
            if lines:
                blocks.append(f"[Лист {sheet.name}]\n" + "\n".join(lines))

        if not blocks:
            raise ExtractionError("No text could be extracted from XLS")
        return blocks
