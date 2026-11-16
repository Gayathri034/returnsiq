"""Load the policy documents and split them into section-level chunks."""
import re
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


@dataclass
class Chunk:
    id: str          # e.g. "01_return_policy#return-window"
    doc_title: str   # e.g. "Northwind Home Goods: Standard Return Policy"
    heading: str     # e.g. "Return window"
    text: str        # section body

    @property
    def full_text(self) -> str:
        return f"{self.heading}. {self.text}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """One chunk per '## ' section. Section-level chunks keep each policy
    rule together, which makes retrieval and citations precise."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        doc_title, heading, body = path.stem, None, []

        def flush():
            if heading and body:
                text = " ".join(" ".join(body).split())
                chunks.append(Chunk(f"{path.stem}#{_slug(heading)}", doc_title, heading, text))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                heading, body = line[3:].strip(), []
            elif line.startswith(">"):
                continue  # disclaimer lines are not policy content
            elif heading:
                body.append(line.strip())
        flush()
    if not chunks:
        raise FileNotFoundError(f"No policy documents found in {docs_dir}")
    return chunks


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
