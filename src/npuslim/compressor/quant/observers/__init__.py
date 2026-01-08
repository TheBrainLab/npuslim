from .base_observer import ParentObserver
from .ptq_observer import PTQObserver
from .abs_max_weight import AbsMaxChannelWiseWeightObserver
from .abs_max_activation import (
    AbsmaxPertensorObserver,
    AbsMaxTokenWiseActObserver,
    AbsmaxPerchannelObserver,
)


__all__ = [
    "ParentObserver",
    "PTQObserver",
    "AbsMaxChannelWiseWeightObserver",
    "AbsmaxPertensorObserver",
    "AbsMaxTokenWiseActObserver",
    "AbsmaxPerchannelObserver",
]


ACT_OBSERVERS_CLASS = {
    # "per-tensor": AbsmaxPertensorObserver,
    "per-channel": AbsmaxPerchannelObserver,
}
WEIGHT_OBSERVERS_CLASS = {
    # "per-tensor": AbsmaxPertensorObserver,
    "per-channel": AbsMaxChannelWiseWeightObserver,
    # "per-group": AbsMaxGroupWiseWeightObserver,
}