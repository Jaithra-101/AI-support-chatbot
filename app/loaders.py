"""Turn uploaded files into (page_number, text) pairs."""
import io

from pypdf import PdfReader

SUPPORTED = (".pdf", ".txt", ".md")


def load_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if name.endswith((".txt", ".md")):
        return [(1, data.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type. Use one of: {', '.join(SUPPORTED)}")
