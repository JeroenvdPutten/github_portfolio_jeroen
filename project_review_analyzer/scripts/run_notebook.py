"""Execute the learning notebook with the tools container's Python kernel."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks" / "01_evaluation.ipynb", as_version=4)
    NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
    output = root / "reports" / "01_evaluation.executed.ipynb"
    output.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Executed notebook saved to {output}")


if __name__ == "__main__":
    main()
