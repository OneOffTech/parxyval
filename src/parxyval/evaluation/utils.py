from difflib import SequenceMatcher
from typing import Optional, List, Dict

from parxy_core.models import BoundingBox, TextBlock, Page

HEADING_LABELS = ['heading', 'title', 'section-header']
HEADING_MATCHER_TEXT_SIMILARITY_THRESHOLD = 0.9


def bbox_iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.

    Args:
        b1 : BoundingBox
            The first bounding box.
        b2 : BoundingBox
            The second bounding box.

    Returns
    -------
    float
        IoU value in [0, 1].
    """
    x_left = max(b1.x0, b2.x0)
    y_top = max(b1.y0, b2.y0)
    x_right = min(b1.x1, b2.x1)
    y_bottom = min(b1.y1, b2.y1)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1.x1 - b1.x0) * (b1.y1 - b1.y0)
    area2 = (b2.x1 - b2.x0) * (b2.y1 - b2.y0)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def match_bboxes(
    true_doc: Page,
    pred_doc: Page,
    pivot: Optional[str] = None,
) -> List[Dict[str, List]]:
    """
    Function adapted from
    [docling-core](https://github.com/docling-project/docling-core/blob/main/docling_core/types/doc/base.py).

    Args:
        pivot : str
                It must be either None or one of ["true", "pred"].
                If it is None the pivot is the document with less bboxes.

    Returns
    --------
    List with matchings for the bboxes/text between the true and pred docs.
    - Each list item is a dict with the matching between the true and the pred bboxes.
    - Each dict has the keys:
      "true_bboxes": List[BoundingBox] of true bboxes
      "true_tokens": List[str] of the tokenized text contained in the true_bboxes
      "pred_bboxes": List[BoundingBox] of pred bboxes
      "pred_tokens": List[str] of the tokenized text contained in the pred_bboxes
    """
    if pivot is not None:
        assert pivot in ['true', 'pred']

    # Collect bboxes from both documents (true, pred)
    bboxes: Dict[str, List[BoundingBox]] = {'true': [], 'pred': []}
    texts: Dict[str, List[str]] = {'true': [], 'pred': []}
    for doc_key, doc in {'true': true_doc, 'pred': pred_doc}.items():
        for doc_item in doc.blocks:
            if not isinstance(doc_item, TextBlock):
                continue
            bboxes[doc_key].append(doc_item.bbox)
            texts[doc_key].append(doc_item.text)

    # Decide which document is the pivot
    if pivot is None:
        pivot = 'true' if len(bboxes['true']) <= len(bboxes['pred']) else 'pred'
    other = 'pred' if pivot == 'true' else 'true'

    # Map the "pivot" bboxes to the "other" bboxes
    # Keys: the indices from bboxes[pivot]. Each value: list with indices from bboxes[other]
    pivot_mappings: Dict[int, List[int]] = {}
    all_other_ids = set()
    for other_id, other_bbox in enumerate(bboxes[other]):
        max_iou = None
        max_pivot_id = None
        for pivot_id, pivot_bbox in enumerate(bboxes[pivot]):
            iou = bbox_iou(pivot_bbox, other_bbox)
            if max_iou is None or max_iou < iou:
                max_iou = iou
                max_pivot_id = pivot_id
        if max_iou is not None and max_pivot_id is not None:
            if max_pivot_id not in pivot_mappings:
                pivot_mappings[max_pivot_id] = []
            pivot_mappings[max_pivot_id].append(other_id)
            all_other_ids.add(other_id)

    # Collect the unmatched true bboxes
    orphan_trues: List[int] = []
    for true_id in range(len(bboxes['true'])):
        if pivot == 'true':
            if true_id not in pivot_mappings:
                orphan_trues.append(true_id)
        else:
            if true_id not in all_other_ids:
                orphan_trues.append(true_id)

    # Create mapping for the text of the matched bboxes
    # Each dict has the keys:
    #  "true_bboxes": List[BoundingBox] of true bboxes
    #  "true_tokens": List[str] of the tokenized text contained in the true_bboxes
    #  "pred_bboxes": List[BoundingBox] of pred bboxes
    #  "pred_tokens": List[str] of the tokenized text contained in the pred_bboxes
    matches: list[Dict[str, list]] = []
    for pivot_id, list_other_ids in pivot_mappings.items():
        pivot_bboxes = [bboxes[pivot][pivot_id]]
        pivot_text = texts[pivot][pivot_id]
        pivot_tokens = pivot_text.split()
        other_tokens = []
        other_bboxes = []
        for other_id in list_other_ids:
            other_text = texts[other][other_id]
            other_tokens.extend(other_text.split())
            other_bboxes.append(bboxes[other][other_id])

        matches.append(
            {
                f'{pivot}_bboxes': pivot_bboxes,
                f'{pivot}_tokens': pivot_tokens,
                f'{other}_bboxes': other_bboxes,
                f'{other}_tokens': other_tokens,
            }
        )

    # Add the orphans_true inside the matches
    for orphan_true_id in orphan_trues:
        orphan_bboxes = [bboxes['true'][orphan_true_id]]
        orphan_text = texts['true'][orphan_true_id]
        orphan_tokens = orphan_text.split()
        matches.append(
            {
                'true_bboxes': orphan_bboxes,
                'true_tokens': orphan_tokens,
                'pred_bboxes': [],
                'pred_tokens': [],
            }
        )

    return matches


def text_block_match(
    block1: TextBlock,
    block2: TextBlock,
    text_thresh: float = 0.8,
    iou_thresh: float = 0.5,
) -> bool:
    """Check if two text blocks match based on text similarity or bounding box IoU.

    Two text blocks match if their text similarity is greater than to `text_thresh`
    or their bounding box overlap ratio is greater than `iou_thresh`.

    Args:
        block1 : TextBlock
            The first text block.
        block2 : TextBlock
            The second text block.
        text_thresh : float, optional
            The minimum text similarity threshold. Default to 0.8.
        iou_thresh : float, optional
            The minimum IoU threshold. Default to 0.5.

    Returns
    -------
    bool
        True if blocks match, False otherwise.
    """
    if block1.page != block2.page:
        return False
    text_sim = text_similarity(block1.text, block2.text)
    iou = bbox_iou(block1.bbox, block2.bbox) if block1.bbox and block2.bbox else 0.0
    return text_sim >= text_thresh or iou >= iou_thresh


def text_similarity(a: str, b: str) -> float:
    """Compute SequenceMatcher ratio between two strings.

    Args:
        a : str
            The first string.
        b : str
            The second string.

    Returns
    -------
    float
        The similarity ratio in [0, 1].
    """
    return SequenceMatcher(None, a, b).ratio()
