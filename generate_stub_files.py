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
    if not content.endswith("\n"):
        content += "\n"

    # Regular expression to match autosummary blocks
    # Matches: .. autosummary::
    #          [indented options/text]
    #          [blank line]
    #          [indented entries]
    autosummary_pattern = re.compile(
        r"\.\.\s+autosummary::[^\n]*\n"
        r"(?P<options>(?:[ \t]+[^\n]*\n)*)\n"
        r"(?P<entries>(?:(?:[ \t]+[^\n]*\n)|(?:[ \t]*\n))*)",
        re.MULTILINE
    )

    for match in autosummary_pattern.finditer(content):
        options = match.group("options")
        entries_str = match.group("entries")

        toctree_match = re.search(r"^[ \t]+:toctree:[ \t]+(\S+)", options, re.MULTILINE)
        if not toctree_match:
            continue

        toctree = toctree_match.group(1).strip("/")
        toctree = toctree.split("/")[-1]  # e.g. '../method' -> 'method'

        if toctree == "method":
            directive = "automethod"
            out_dir = OUT_DIR / "method"
        elif toctree == "attribute":
            directive = "autoattribute"
            out_dir = OUT_DIR / "attribute"
        elif toctree == "function":
            directive = "autofunction"
            out_dir = OUT_DIR / "function"
        else:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract fully qualified names, ignoring comments
        entries = []
        for line in entries_str.splitlines():
            line = line.strip()
            if not line or line.startswith(".."):
                continue
            # Remove leading ~ if present
            if line.startswith("~"):
                line = line[1:]
            # Ensure it looks like a cf object
            if not line.startswith("cf."):
                continue
            
            entry_match = re.match(r"([\w\.\_]+)", line)
            if entry_match:
                entries.append(entry_match.group(1))

        for entry in entries:
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


# Run over all class rst files and function.rst
if __name__ == "__main__":
    rst_files = list(SOURCE_DIR.glob("*.rst"))
    
    function_rst = OUT_DIR / "function.rst"
    if function_rst.exists():
        rst_files.append(function_rst)

    for rst_file in sorted(rst_files):
        print(f"Parent rst file: {rst_file}")
        generate_stub_files(rst_file)
