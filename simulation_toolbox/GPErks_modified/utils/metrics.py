import torch
from torchmetrics import Metric


def get_metric_name(metric: Metric) -> str:
    return metric.__class__.__name__


def IndependentStandardError(y_true, y_pred_mean, y_pred_std):
    ise = torch.abs(y_true - y_pred_mean) / y_pred_std
    return 100.0 * len(torch.where(ise < 2.0)[0]) / len(ise)


class IndependentStandardErrorMetric(Metric):
    def __init__(self):
        super().__init__()

    def update(self, y_true, y_pred_mean, y_pred_std):
        # Compute the ISE for each sample
        ise = torch.abs(y_true - y_pred_mean) / y_pred_std
        self.ise = ise

    def compute(self):
        # Return the percentage of predictions where error is less than twice the std
        ise_less_than_2 = torch.sum(self.ise < 2.0).float()
        total_samples = len(self.ise)
        return 100.0 * ise_less_than_2 / total_samples


def MAPE(y, y_pred):
    n_samples = y.size()[0]
    y_c = y.detach().clone()
    l = torch.where(y == 0)[0].tolist()
    if l:
        nl = list(set(range(n_samples)) - set(l))
        correction = torch.min(torch.abs(y[nl]))
        for idx in l:
            y_c[idx] = correction

    abs_rel_errors = torch.abs((y - y_pred)) / torch.abs(y_c)
    sum_of_abs_rel_errors = torch.sum(abs_rel_errors).item()
    return 100.0 * sum_of_abs_rel_errors / n_samples
