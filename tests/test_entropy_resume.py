"""Restored training must not apply the schedule's initial coefficient again."""
from collections import defaultdict
from types import SimpleNamespace

import pytest
from ray.rllib.utils.schedules.scheduler import Scheduler

from ngllib_agent.train import _restore_entropy_schedule


@pytest.mark.parametrize('steps,expected', [(0, .2), (900000, .05), (3000000, .01), (17000000, .01)])
def test_lazy_entropy_scheduler_is_restored_before_first_update(steps, expected):
    schedulers = defaultdict(lambda: Scheduler(
        fixed_value_or_schedule=[[0, .2], [900000, .05], [3000000, .01]], framework='torch', device='cpu'))
    learner = SimpleNamespace(module={'default_policy': object()}, entropy_coeff_schedulers_per_module=schedulers)
    algo = SimpleNamespace(learner_group=SimpleNamespace(_learner=learner))
    assert not schedulers
    restored = _restore_entropy_schedule(algo, steps)
    assert restored['default_policy'] == pytest.approx(expected)
    assert float(schedulers['default_policy'].get_current_value()) == pytest.approx(expected)
