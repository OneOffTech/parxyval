import pymupdf

from parxy_core.models.models import Document, TextBlock, Line, Span, estimate_lines_from_block


class ConvertToPDF:
    """Class to convert a Document object to a PDF file using PyMuPDF or ReportLab."""

    def __init__(
            self,
            font_name: str = "Helvetica",
            font_size: float = 10,
            page_width: float = 595,
            page_height: float = 842,
    ):
        """Initialize the PDF converter with default font and page specifications.

        Args:
            font_name : str, optional
                Font name to use. Default to "Helvetica".
            font_size : float, optional
                Font size to use. Default to 10.
            page_width : float, optional
                Default page width in points. Default to 595.
            page_height : float, optional
                Default page height in points. Default to 842.
        """
        self.specs = {
            "font_name": font_name,
            "font_size": font_size,
            "page_width": page_width,
            "page_height": page_height,
        }

    def build(
            self,
            doc: Document,
            output_filepath: str,
            driver: str = "pymupdf",
            estimate_missing_lines: bool = False,
    ):
        """Build a PDF file from a Document object.

        Args:
            doc : Document
                THe document to convert.
            output_filepath : str
                The path to save the output PDF.
            driver : str, optional
                The PDF library to use ("pymupdf" or "reportlab"). Default to "pymupdf".
            estimate_missing_lines : bool, optional
                If True, estimates missing lines from text blocks. Default to False.

        Raises
        -------
        NotImplementedError
            If an unsupported driver is specified.
        """
        if estimate_missing_lines:
            # Try to estimate missing lines from blocks
            for page in doc.pages:
                for i, block in enumerate(page.blocks):
                    page.blocks[i] = estimate_lines_from_block(block)

        if driver == "pymupdf":
            self.build_with_pymupdf(doc, output_filepath)
        elif driver == "reportlab":
            self.build_with_reportlab(doc, output_filepath)
        else:
            raise NotImplementedError

    def build_with_pymupdf(
            self,
            doc: Document,
            output_filepath: str
    ):
        """Build a PDF using the PyMuPDF library.

        Args:
            doc : Document
                The document to convert.
            output_filepath : str
                The path to save the PDF.
        """
        pdf = pymupdf.open()

        for page_data in doc.pages:
            # Create a new blank page with given width and height
            width = page_data.width or self.specs["page_width"]
            height = page_data.height or self.specs["page_height"]
            page = pdf.new_page(pno=-1, width=width, height=height)

            if not page_data.blocks:
                continue

            for block in page_data.blocks:
                if isinstance(block, TextBlock):
                    if block.lines:
                        for line in block.lines:
                            if line.spans:
                                # Insert lines if possible
                                for span in line.spans:
                                    self._insert_element_with_pymupdf(page, span)
                            else:
                                # Otherwise, insert the entire line directly
                                self._insert_element_with_pymupdf(page, line)
                    else:
                        # Otherwise, insert the entire line directly
                        self._insert_element_with_pymupdf(page, block)

        pdf.save(str(output_filepath))
        pdf.close()

    def build_with_reportlab(
            self,
            doc: Document,
            output_filepath: str
    ):
        """Build a PDF using the ReportLab library.

        Args:
            doc : Document
                The document to convert.
            output_filepath : str
                The path to save the PDF.

        Raises
        -------
        ImportError
            If the `reportlab` package is not installed.
        """
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            raise ImportError("The `reportlab` package is not installed. Install it with `pip install reportlab`.")
        c = canvas.Canvas(output_filepath)

        for page_data in doc.pages:
            width = page_data.width or self.specs["page_width"]
            height = page_data.height or self.specs["page_height"]
            c.setPageSize((width, height))

            if not page_data.blocks:
                continue

            for block in page_data.blocks:
                if isinstance(block, TextBlock):
                    if block.lines:
                        for line in block.lines:
                            if line.spans:
                                # Insert lines if possible
                                for span in line.spans:
                                    self._insert_element_with_reportlab(c, span, height)
                            else:
                                # Otherwise, insert the entire line directly
                                self._insert_element_with_reportlab(c, line, height)
                    else:
                        # Otherwise, insert the entire line directly
                        self._insert_element_with_reportlab(c, block, height)
            c.showPage()

        if c:
            c.save()

    def _insert_element_with_pymupdf(
            self,
            page: pymupdf.Page,
            element: TextBlock | Line | Span
    ):
        """Insert a text element into a PyMuPDF page.

        Args:
            page : pymupdf.Page
                The page to insert the element into.
            element : TextBlock | Line | Span
                The text element to insert.
        """
        if not element.text or not element.bbox:
            return
        x0 = element.bbox.x0
        y0 = element.bbox.y0
        font_size = element.style.font_size if element.style else None
        # Draw text in the top-left corner of the bbox
        page.insert_text(
            point=(x0, y0),
            text=element.text,
            fontsize=font_size or self.specs["font_size"],
            fontname=self.specs["font_name"],
            color=(0, 0, 0)
        )

    def _insert_element_with_reportlab(
            self,
            c: "canvas.Canvas",
            element: TextBlock | Line | Span,
            page_height: float
    ):
        """Insert a text element into a ReportLab canvas.

        Args:
            c : canvas.Canvas
                ReportLab canvas.
            element : TextBlock | Line | Span)
                The text element to insert.
            page_height : float
                Height of the page to adjust Y-coordinate.
        """
        if not element.text or not element.bbox:
            return
        x0 = element.bbox.x0
        y0 = page_height - element.bbox.y0
        font_size = element.style.font_size if element.style else None
        c.setFont(self.specs["font_name"], font_size or self.specs["font_size"])
        c.drawString(x0, y0, element.text)
