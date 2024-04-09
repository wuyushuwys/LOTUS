from collections import namedtuple

from .meter import AverageMeter, TimeMeter, QvalueMeter
from .csv_writer import CSVWriter
from .action_scheduler import ActionScheduler


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def soft_update(target, source, tau):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target.state_dict()
    policy_net_state_dict = source.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
    target.load_state_dict(target_net_state_dict)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())