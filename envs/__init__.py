from gym.envs.registration import register

from envs.micro_tbs import MicroTbs, GameMode

register(
    id='MicroTbs-CollectSimple-v2',
    entry_point='envs.micro_tbs:MicroTbs',
    kwargs={'mode': GameMode.collect_gold_simple_no_terrain()},
)

register(
    id='MicroTbs-CollectWithTerrain-v2',
    entry_point='envs.micro_tbs:MicroTbs',
    kwargs={'mode': GameMode.collect_gold_simple_terrain()},
)

register(
    id='MicroTbs-CollectPartiallyObservable-v3',
    entry_point='envs.micro_tbs:MicroTbs',
    kwargs={'mode': GameMode.collect_gold_partially_observable()},
)
