from parxy_core.models import TextBlock, Document, Page

from parxyval.evaluation.factory import register_metric
from parxyval.evaluation.utils import (
    HEADING_LABELS,
    HEADING_MATCHER_TEXT_SIMILARITY_THRESHOLD,
    text_similarity,
)


@register_metric('headings_matcher')
def headings_matcher(doc: Document, gt_doc: Document) -> dict:
    """Evaluate heading detection metrics between predicted and ground truth documents.

    Args:
        doc : Document
            The predicted document.
        gt_doc : Document
            The ground truth document.

    Returns
    -------
    dict
        A dictionary with keys: `heading_precision`, `heading_recall`, `heading_f1_score`.
    """
    correct = 0
    total_gt, total_pred = 0, 0
    for page_doc, page_gt in zip(doc.pages, gt_doc.pages):
        gt_headings = _extract_headings(page_gt)
        pred_headings = _extract_headings(page_doc)
        matched_preds = set()

        for gt in gt_headings:
            for idx, pred in enumerate(pred_headings):
                if idx in matched_preds:
                    continue
                if (
                    text_similarity(gt.text.lower(), pred.text.lower())
                    > HEADING_MATCHER_TEXT_SIMILARITY_THRESHOLD
                ):
                    correct += 1
                    matched_preds.add(idx)
                    break

        total_gt += len(gt_headings)
        total_pred += len(pred_headings)

    precision = correct / total_pred if total_pred else 0.0
    recall = correct / total_gt if total_gt else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        'heading_precision': precision,
        'heading_recall': recall,
        'heading_f1_score': f1,
    }


def _extract_headings(page: Page) -> list[TextBlock]:
    """Extract heading text blocks from a page.

    Args:
        page : Page
            The page to extract headings from.

    Returns
    -------
    list[TextBlock]
        The list of text blocks labeled as headings or titles.
    """
    return [
        block
        for block in page.blocks or []
        if isinstance(block, TextBlock)
        and isinstance(block.category, str)
        and block.category.lower() in HEADING_LABELS
    ]
