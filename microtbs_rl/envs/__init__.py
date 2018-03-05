from gym.envs.registration import register

from microtbs_rl.envs.micro_tbs import MicroTbs, GameMode


COLLECT_SIMPLE_LATEST = 'MicroTbs-CollectSimple-v2'
register(
    id=COLLECT_SIMPLE_LATEST,
    entry_point='microtbs_rl.envs.micro_tbs:MicroTbs',
    kwargs={'mode': GameMode.collect_gold_simple_no_terrain()},
)

COLLECT_WITH_TERRAIN_LATEST = 'MicroTbs-CollectWithTerrain-v2'
register(
    id=COLLECT_WITH_TERRAIN_LATEST,
    entry_point='microtbs_rl.envs.micro_tbs:MicroTbs',
    kwargs={'mode': GameMode.collect_gold_simple_terrain()},
)

COLLECT_PARTIALLY_OBSERVABLE_LATEST = 'MicroTbs-CollectPartiallyObservable-v3'
register(
    id=COLLECT_PARTIALLY_OBSERVABLE_LATEST,
    entry_point='microtbs_rl.envs.micro_tbs:MicroTbs',
    kwargs={'mode': GameMode.collect_gold_partially_observable()},
)
