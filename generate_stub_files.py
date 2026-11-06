"""It will automatically detect whether an entry falls under

.. rubric:: Methods or .. rubric:: Attributes

select automethod or autoattribute, and write out the exact .rst file
needed.

"""

import re
import sys
from pathlib import Path

if len(sys.argv) > 1:
    SOURCE_DIR = Path(sys.argv[1])
else:
    SOURCE_DIR = Path("source/class")

OUT_DIR = SOURCE_DIR.parent


def generate_stub_files(rst_path: Path):
    content = rst_path.read_text(encoding="utf-8")

    # Regular expression to match rubric sections and their
    # autosummary entries
    #
    # Matches: .. rubric:: (Methods|Attributes) ... .. autosummary:: ... [entries]
    section_pattern = re.compile(
        r"\.\.\s+rubric::\s+(Methods|Attributes)\s*?\n"
        r"(?:[^\n]*\n)*?"  # skip options until entries
        r"\.\.\s+autosummary::.*?\n\n"
        r"((?:\s+~[\w\.\_]+\n)+)",
        re.MULTILINE | re.DOTALL,
    )

    for match in section_pattern.finditer(content):
        member_type = match.group(1).lower()  # 'methods' or 'attributes'
        raw_entries = match.group(2)

        # Extract fully qualified names (e.g.,
        # 'cf.AuxiliaryCoordinate.del_properties')
        entries = re.findall(r"~\s*([\w\.\_]+)", raw_entries)

        for entry in entries:
            # Determine directive (automethod vs autoattribute)
            if member_type == "methods":
                directive = "automethod"
            else:
                directive = "autoattribute"

            # Determine output directory (method/ or attribute/)
            if member_type == "methods":
                out_dir = OUT_DIR / "method"
            else:
                out_dir = OUT_DIR / "attribute"

            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f"{entry}.rst"

            # Format underline to match title length
            underline = "=" * len(entry)

            stub_content = (
                f"{entry}\n"
                f"{underline}\n\n"
                f".. currentmodule:: {entry.split('.')[0]}\n"
                f".. default-role:: obj\n\n"
                f".. {directive}:: {entry}\n"
            )

            out_file.write_text(stub_content, encoding="utf-8")
            print(f"    Generated: {out_file}")


# Run over all class rst files
if __name__ == "__main__":
    for rst_file in sorted(SOURCE_DIR.glob("*.rst")):
        print(f"Parent rst file: {rst_file}")
        generate_stub_files(rst_file)
