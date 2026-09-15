"""One-off: extract title/abstract/intro text from paper/*.pdf for trend analysis."""
import re
import sys
from pathlib import Path

from pypdf import PdfReader

PAPER_DIR = Path("/root/autodl-tmp/multimodal_attack_project/paper")


def clean(t: str) -> str:
    t = t.replace("\u00ad", "")
    t = re.sub(r"-\n", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def main() -> None:
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for pdf in sorted(PAPER_DIR.glob("*.pdf")):
        if only and not any(o.lower() in pdf.name.lower() for o in only):
            continue
        print("=" * 100)
        print("FILE:", pdf.name)
        try:
            reader = PdfReader(str(pdf))
            n = len(reader.pages)
            print("PAGES:", n)
            # first 3 pages usually hold title/abstract/intro
            text = []
            for i in range(min(3, n)):
                text.append(reader.pages[i].extract_text() or "")
            body = clean("\n".join(text))
            print(body[:4000])
        except Exception as e:  # noqa: BLE001
            print("ERROR:", e)
        print()


if __name__ == "__main__":
    main()
