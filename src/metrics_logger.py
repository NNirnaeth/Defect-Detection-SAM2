import os
import csv
from datetime import datetime

class MetricsLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self._init_file()

    def _init_file(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timestamp", "mode", "dataset", "steps", "model_name",
                    "IoU", "IoU@50", "IoU@75", "IoU@95",
                    "Precision", "Recall", "F1", "Benevolente",
                    "seg_loss", "score_loss", "total_loss"
                ])

    def log(self, mode, dataset, steps, model_name,
            iou=None, iou50=None, iou75=None, iou95=None,
            precision=None, recall=None, f1=None, benevolente=None,
            seg_loss=None, score_loss=None, total_loss=None):

        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(), mode, dataset, steps, model_name,
                round(iou, 4) if iou is not None else "",
                round(iou50 * 100, 2) if iou50 is not None else "",
                round(iou75 * 100, 2) if iou75 is not None else "",
                round(iou95 * 100, 2) if iou95 is not None else "",
                round(precision, 4) if precision is not None else "",
                round(recall, 4) if recall is not None else "",
                round(f1, 4) if f1 is not None else "",
                round(benevolente * 100, 2) if benevolente is not None else "",
                round(seg_loss, 4) if seg_loss is not None else "",
                round(score_loss, 4) if score_loss is not None else "",
                round(total_loss, 4) if total_loss is not None else ""
            ])

