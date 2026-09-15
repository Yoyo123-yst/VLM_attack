"""One-off: extract just the abstract region of each paper for trend analysis."""
import re
from pathlib import Path

from pypdf import PdfReader

PAPER_DIR = Path("/root/autodl-tmp/multimodal_attack_project/paper")
OUT = Path("/root/autodl-tmp/multimodal_attack_project/outputs/_paper_abstracts.md")

STOP = re.compile(
    r"\n\s*(?:1\.?\s*)?(?:I\.\s*)?(?:INTRODUCTION|Introduction|1 Introduction|"
    r"Index Terms|CCS Concepts|Keywords|I\. INTRODUCTION)\b"
)


def clean(t: str) -> str:
    t = t.replace("\u00ad", "")
    t = re.sub(r"-\n", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def harvest(name: str) -> str:
    reader = PdfReader(str(PAPER_DIR / name))
    text = "\n".join((reader.pages[i].extract_text() or "") for i in range(min(4, len(reader.pages))))
    text = clean(text)
    m = re.search(r"(?:Abstract|ABSTRACT|A B S T R A C T)[\s:—-]*", text)
    head = text[:2000]
    if m:
        start = m.end()
        tail = text[start:start + 3000]
        s = STOP.search(tail)
        if s:
            tail = tail[: s.start()]
        return f"{head}\n\n>>> ABSTRACT:\n{tail}"
    return head


def main() -> None:
    parts = ["# Paper abstracts harvest\n"]
    for pdf in sorted(PAPER_DIR.glob("*.pdf")):
        try:
            body = harvest(pdf.name)
        except Exception as e:  # noqa: BLE001
            body = f"ERROR: {e}"
        parts.append(f"\n\n## {pdf.name}\n\n```\n{body}\n```\n")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
