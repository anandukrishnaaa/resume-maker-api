"""
Simple script to generate PDF from a Typst file using RenderCV
"""

from pathlib import Path
import rendercv.renderer as renderer


def generate_pdf_from_typst(typst_file_path: str) -> Path:
    """
    Generate a PDF from a Typst file.

    Args:
        typst_file_path: Path to the .typ file

    Returns:
        Path object pointing to the generated PDF

    Raises:
        FileNotFoundError: If the typst file doesn't exist
    """
    typst_path = Path(typst_file_path)

    if not typst_path.exists():
        raise FileNotFoundError(f"Typst file not found: {typst_file_path}")

    if typst_path.suffix != ".typ":
        raise ValueError("File must have .typ extension")

    print(f"Rendering PDF from: {typst_path}")

    # Use RenderCV's renderer to convert Typst to PDF
    pdf_path = renderer.render_a_pdf_from_typst(typst_path)

    print(f"PDF generated at: {pdf_path}")

    return pdf_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_pdf.py <path_to_typst_file>")
        print("Example: python generate_pdf.py output/task_1/John_Doe_CV.typ")
        sys.exit(1)

    typst_file = sys.argv[1]

    try:
        pdf_path = generate_pdf_from_typst(typst_file)
        print(f"\n✅ Success! PDF available at: {pdf_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
