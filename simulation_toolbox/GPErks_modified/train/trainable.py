from abc import ABCMeta, abstractmethod

from GPErks_modified.train.early_stop import EarlyStoppingCriterion
from GPErks_modified.train.snapshot import SnapshottingCriterion


class Trainable(metaclass=ABCMeta):
    @abstractmethod
    def train(
        self,
        optimizer,
        early_stopping_criterion: EarlyStoppingCriterion,
        snapshotting_criterion: SnapshottingCriterion,
    ):
        pass
