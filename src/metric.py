import cupy as cp


class Metric:
    @staticmethod
    def accuracy(y_true, y_pred):
        """Computes multi-class accuracy."""
        y_true_labels = cp.argmax(
            y_true, axis=1)  # Convert one-hot to class labels
        y_pred_labels = cp.argmax(y_pred, axis=1)
        return cp.mean(y_true_labels == y_pred_labels)

    @staticmethod
    def precision(y_true, y_pred, average='macro'):
        """Computes precision for multi-class classification."""
        y_true_labels = cp.argmax(y_true, axis=1)
        y_pred_labels = cp.argmax(y_pred, axis=1)
        num_classes = y_true.shape[1]

        precisions = []
        for c in range(num_classes):
            tp = cp.sum((y_pred_labels == c) & (y_true_labels == c))
            fp = cp.sum((y_pred_labels == c) & (y_true_labels != c))
            precision = tp / (tp + fp + 1e-12)  # Avoid division by zero
            precisions.append(precision)

        if average == 'macro':
            return cp.mean(cp.array(precisions))
        return precisions  # Return per-class precision

    @staticmethod
    def recall(y_true, y_pred, average='macro'):
        """Computes recall for multi-class classification."""
        y_true_labels = cp.argmax(y_true, axis=1)
        y_pred_labels = cp.argmax(y_pred, axis=1)
        num_classes = y_true.shape[1]

        recalls = []
        for c in range(num_classes):
            tp = cp.sum((y_pred_labels == c) & (y_true_labels == c))
            fn = cp.sum((y_pred_labels != c) & (y_true_labels == c))
            recall = tp / (tp + fn + 1e-12)  # Avoid division by zero
            recalls.append(recall)

        if average == 'macro':
            return cp.mean(cp.array(recalls))
        return recalls  # Return per-class recall

    @staticmethod
    def f1_score(y_true, y_pred, average='macro'):
        """Computes F1-score for multi-class classification."""
        precision = Metric.precision(y_true, y_pred, average=None)
        recall = Metric.recall(y_true, y_pred, average=None)

        f1_scores = []
        for p, r in zip(precision, recall):
            f1 = 2 * (p * r) / (p + r + 1e-12)  # Avoid division by zero
            f1_scores.append(f1)

        if average == 'macro':
            return cp.mean(cp.array(f1_scores))
        return f1_scores  # Return per-class F1-score
