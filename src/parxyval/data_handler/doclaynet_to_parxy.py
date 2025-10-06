from typing import List, Dict, Any

from parxy_core.models import Document, TextBlock, BoundingBox, Style, Page, Line

category_mapping = {
    "1": "Caption",
    "2": "Footnote",
    "3": "Formula",
    "4": "List-item",
    "5": "Page-footer",
    "6": "Page-header",
    "7": "Picture",
    "8": "Section-header",
    "9": "Table",
    "10": "Text",
    "11": "Title",
}


def coco_to_parxy(json_data: Dict[str, Any]) -> Document:
    """
    Convert the given JSON (representing a PDF page in COCO format) into a Document object
    where each cell is represented as a TextBlock.
    """
    # Extract metadata
    metadata = json_data.get("metadata", {})
    page_no = metadata.get("page_no", 1)
    width = metadata.get("coco_width")
    height = metadata.get("coco_height")

    text_blocks: List[TextBlock] = []
    full_page_text = []

    for cell in json_data.get("cells", []):
        # Extract bbox and convert to BoundingBox
        bbox_values = cell.get("bbox", [0, 0, 0, 0])
        bbox = BoundingBox(
            x0=bbox_values[0],
            y0=bbox_values[1],
            x1=bbox_values[0] + bbox_values[2],
            y1=bbox_values[1] + bbox_values[3],
        )

        # Extract font style
        font = cell.get("font", {})
        style = Style(
            font_name=font.get("name"),
            # font_size=font.get("size"),
            color=str(tuple(font.get("color", []))) if font.get("color") else None,
        )

        text = cell.get("text", "")
        full_page_text.append(text)

        text_block = TextBlock(
            type="text",
            bbox=bbox,
            page=page_no,
            text=text,
            style=style,
            source_data=cell
        )
        text_blocks.append(text_block)

    page = Page(
        number=page_no,
        width=width,
        height=height,
        blocks=text_blocks,
        text=" ".join(full_page_text),
        source_data=json_data
    )

    document = Document(
        filename=metadata.get("original_filename"),
        pages=[page],
        source_data=json_data
    )

    return document


def doclaynet_v12_to_parxy(
        pdf_cells: List[List[Dict[str, Any]]],
        metadata: Dict[str, Any],
        category_id: List[int],
) -> Document:
    """
    Convert the given JSON representation of a PDF page into a Document model.
    """
    text_blocks = []
    all_page_texts = []

    for block_data, category in zip(pdf_cells, category_id):
        lines = []
        block_texts = []
        block_bboxes = []

        for line_data in block_data:
            # Extract bounding box
            bbox_values = line_data.get("bbox", [0, 0, 0, 0])
            bbox = BoundingBox(
                x0=bbox_values[0],
                y0=bbox_values[1],
                x1=bbox_values[0] + bbox_values[2],
                y1=bbox_values[1] + bbox_values[3],
            )

            # Extract style
            font = line_data.get("font", {})
            color_rgba = font.get("color", [0, 0, 0, 255])
            if color_rgba is None:
                color_rgba = [0, 0, 0, 255]
            color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgba[:3])
            alpha = color_rgba[3] if len(color_rgba) > 3 else None
            style = Style(
                font_name=font.get("name"),
                # font_size=font.get("size"),
                color=color_hex,
                alpha=alpha
            )

            text = line_data.get("text", "")
            block_texts.append(text)
            block_bboxes.append(bbox)

            line = Line(
                text=text,
                bbox=bbox,
                style=style,
                page=metadata["page_no"]
            )

            lines.append(line)

        # Compute block bounding box (union of all lines)
        if block_bboxes:
            x0 = min(bb.x0 for bb in block_bboxes)
            y0 = min(bb.y0 for bb in block_bboxes)
            x1 = max(bb.x1 for bb in block_bboxes)
            y1 = max(bb.y1 for bb in block_bboxes)
            block_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
        else:
            block_bbox = None

        block_text = "".join(block_texts)
        all_page_texts.append(block_text)

        text_block = TextBlock(
            type="text",
            bbox=block_bbox,
            category=category_mapping[str(category)],
            page=metadata["page_no"],
            text=block_text,
            lines=lines
        )
        text_blocks.append(text_block)

    page_text = "\n".join(all_page_texts)
    page = Page(
        number=metadata["page_no"],
        width=metadata["original_width"],
        height=metadata["original_height"],
        blocks=text_blocks,
        text=page_text
    )

    return Document(
        filename=metadata["original_filename"],
        pages=[page],
        source_data=metadata
    )
