import pytest
from pydantic import ValidationError

from surprise_loop.world import (
    Action,
    Event,
    Outcome,
    PhaseKind,
    PhaseSpec,
    SelfAdminWorld,
    World,
    WorldConfig,
    WorldDone,
    WorldState,
)


def fr(ratio: int, steps: int) -> PhaseSpec:
    return PhaseSpec(kind=PhaseKind.FIXED_RATIO, steps=steps, ratio=ratio)


def pr(start: int, increment: int, steps: int) -> PhaseSpec:
    return PhaseSpec(
        kind=PhaseKind.PROGRESSIVE_RATIO, steps=steps, pr_start=start, pr_increment=increment
    )


def ext(steps: int) -> PhaseSpec:
    return PhaseSpec(kind=PhaseKind.EXTINCTION, steps=steps)


def make_world(*phases: PhaseSpec, active: str = "press_a", seed: int = 0) -> SelfAdminWorld:
    config = WorldConfig(active_lever=active, reward_magnitude=1.0, schedule=list(phases))
    return SelfAdminWorld(config, seed)


def run(world: SelfAdminWorld, actions: list[Action]) -> list[Outcome]:
    return [world.step(a) for a in actions]


def mixed_script(n: int) -> list[Action]:
    cycle = [Action.PRESS_A, Action.PRESS_A, Action.PRESS_B, Action.EXPLORE, Action.PRESS_A]
    cycle += [Action.REST, Action.PRESS_B, Action.PRESS_A]
    return [cycle[i % len(cycle)] for i in range(n)]


def test_self_admin_world_satisfies_protocol() -> None:
    world: World = make_world(fr(1, 1))
    assert not world.done


def test_fr5_rewards_exactly_presses_5_10_15() -> None:
    world = make_world(fr(5, 19))
    outcomes = run(world, [Action.PRESS_A] * 19)
    rewarded = [i + 1 for i, o in enumerate(outcomes) if o.observation.event == Event.REWARD]
    assert rewarded == [5, 10, 15]
    for i, o in enumerate(outcomes, start=1):
        if i in rewarded:
            assert o.observation.reward == 1.0
            assert o.info.ratio_completed == 5
        else:
            assert o.observation.event == Event.NO_REWARD
            assert o.observation.reward == 0.0
            assert o.info.ratio_completed is None
        assert o.info.ratio_required == 5


def test_fr5_counts_only_active_presses() -> None:
    world = make_world(fr(5, 36))
    script = mixed_script(36)  # 18 active presses
    outcomes = run(world, script)
    active_count = 0
    rewarded_at: list[int] = []
    for action, o in zip(script, outcomes, strict=True):
        if action == Action.PRESS_A:
            active_count += 1
        if o.observation.event == Event.REWARD:
            rewarded_at.append(active_count)
    assert rewarded_at == [5, 10, 15]


def test_inactive_explore_rest_never_reward_and_have_own_event() -> None:
    world = make_world(fr(1, 30))
    expected = {
        Action.PRESS_B: Event.NO_REWARD,
        Action.EXPLORE: Event.EXPLORED,
        Action.REST: Event.RESTED,
    }
    for action in [Action.PRESS_B, Action.EXPLORE, Action.REST] * 10:
        o = world.step(action)
        assert o.observation.event == expected[action]
        assert o.observation.reward == 0.0
        assert o.observation.last_action == action
        assert o.info.ratio_completed is None
    assert world.snapshot().presses_toward_ratio == 0


def test_pr_requirements_1_3_5_7() -> None:
    world = make_world(pr(1, 2, 16))
    outcomes = run(world, [Action.PRESS_A] * 16)
    completed = [o.info.ratio_completed for o in outcomes if o.observation.event == Event.REWARD]
    assert completed == [1, 3, 5, 7]
    rewarded_steps = [o.info.step for o in outcomes if o.observation.event == Event.REWARD]
    assert rewarded_steps == [1, 4, 9, 16]
    assert [o.info.ratio_required for o in outcomes[:5]] == [1, 3, 3, 3, 5]


def test_extinction_never_rewards() -> None:
    world = make_world(ext(50))
    outcomes = run(world, mixed_script(50))
    assert all(o.observation.reward == 0.0 for o in outcomes)
    assert all(o.observation.event != Event.REWARD for o in outcomes)
    assert all(o.info.ratio_required is None for o in outcomes)
    assert all(o.info.ratio_completed is None for o in outcomes)


def test_phase_change_after_steps_resets_counter_and_last_step_names_phase() -> None:
    world = make_world(fr(5, 3), pr(2, 1, 4), ext(2))
    outcomes = run(world, [Action.PRESS_A] * 3)
    assert [o.info.phase_index for o in outcomes] == [0, 0, 0]
    assert outcomes[-1].info.phase_kind == PhaseKind.FIXED_RATIO
    snap = world.snapshot()
    assert snap.phase_index == 1
    assert snap.step_in_phase == 0
    assert snap.presses_toward_ratio == 0
    assert snap.reinforcers_in_phase == 0

    outcomes = run(world, [Action.PRESS_A] * 4)
    assert [o.info.phase_index for o in outcomes] == [1, 1, 1, 1]
    assert outcomes[0].info.phase_kind == PhaseKind.PROGRESSIVE_RATIO
    assert outcomes[-1].info.phase_kind == PhaseKind.PROGRESSIVE_RATIO
    # PR counts from zero after the FR phase: rewarded at its 2nd press, then needs 3.
    assert [o.observation.event for o in outcomes] == [
        Event.NO_REWARD,
        Event.REWARD,
        Event.NO_REWARD,
        Event.NO_REWARD,
    ]
    assert world.snapshot().presses_toward_ratio == 0

    outcomes = run(world, [Action.PRESS_A] * 2)
    assert [o.info.phase_kind for o in outcomes] == [PhaseKind.EXTINCTION] * 2
    assert outcomes[-1].info.phase_index == 2
    assert world.done
    assert world.snapshot().phase_index == 3


def test_active_press_only_for_active_lever() -> None:
    for active, other in [("press_a", Action.PRESS_B), ("press_b", Action.PRESS_A)]:
        world = make_world(fr(2, 20), active=active)
        for action in [Action(active), other, Action.EXPLORE, Action.REST] * 5:
            o = world.step(action)
            assert o.info.active_press == (action == Action(active))


def test_same_seed_and_script_give_identical_outcomes() -> None:
    phases = (fr(3, 20), pr(1, 2, 20), ext(10))
    script = mixed_script(50)
    first = run(make_world(*phases, active="random", seed=7), script)
    second = run(make_world(*phases, active="random", seed=7), script)
    assert first == second


def test_random_active_lever_depends_only_on_seed_and_takes_both_values() -> None:
    def lever(seed: int, phases: tuple[PhaseSpec, ...]) -> Action:
        return make_world(*phases, active="random", seed=seed).snapshot().active_lever

    levers = {lever(seed, (fr(1, 1),)) for seed in range(20)}
    assert levers == {Action.PRESS_A, Action.PRESS_B}
    for seed in range(20):
        assert lever(seed, (fr(1, 1),)) == lever(seed, (ext(9), pr(2, 3, 4)))


def test_fixed_active_lever_is_used() -> None:
    assert make_world(fr(1, 1), active="press_b").snapshot().active_lever == Action.PRESS_B


def test_snapshot_restore_mid_run_continues_identically() -> None:
    phases = (fr(3, 10), pr(1, 2, 15), ext(5))
    config = WorldConfig(active_lever="random", reward_magnitude=2.5, schedule=list(phases))
    script = mixed_script(30)

    uninterrupted = SelfAdminWorld(config, seed=3)
    expected = run(uninterrupted, script)

    interrupted = SelfAdminWorld(config, seed=3)
    head = run(interrupted, script[:13])
    state = interrupted.snapshot()
    restored = SelfAdminWorld.restore(
        config, WorldState.model_validate_json(state.model_dump_json())
    )
    assert restored.observe() == interrupted.observe()
    tail = run(restored, script[13:])

    assert head + tail == expected
    assert restored.snapshot() == uninterrupted.snapshot()
    assert restored.done and uninterrupted.done


def test_observe_start_then_last_observation_including_after_final_step() -> None:
    world = make_world(fr(2, 3))
    start = world.observe()
    assert start.event == Event.START
    assert start.step == 0
    assert start.last_action is None
    assert start.reward == 0.0

    for _ in range(3):
        o = world.step(Action.PRESS_A)
        assert world.observe() == o.observation
        assert o.observation.step == o.info.step
    assert world.done
    assert world.observe().step == 3
    assert world.observe() == o.observation


def test_step_after_schedule_ends_raises_world_done() -> None:
    world = make_world(fr(1, 2))
    run(world, [Action.REST, Action.REST])
    assert world.done
    with pytest.raises(WorldDone):
        world.step(Action.PRESS_A)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": PhaseKind.FIXED_RATIO, "steps": 5},
        {"kind": PhaseKind.FIXED_RATIO, "steps": 5, "ratio": 0},
        {"kind": PhaseKind.FIXED_RATIO, "steps": 5, "ratio": 2, "pr_start": 1},
        {"kind": PhaseKind.FIXED_RATIO, "steps": 5, "ratio": 2, "pr_increment": 1},
        {"kind": PhaseKind.PROGRESSIVE_RATIO, "steps": 5},
        {"kind": PhaseKind.PROGRESSIVE_RATIO, "steps": 5, "pr_start": 1},
        {"kind": PhaseKind.PROGRESSIVE_RATIO, "steps": 5, "pr_increment": 1},
        {"kind": PhaseKind.PROGRESSIVE_RATIO, "steps": 5, "pr_start": 0, "pr_increment": 1},
        {"kind": PhaseKind.PROGRESSIVE_RATIO, "steps": 5, "pr_start": 1, "pr_increment": -1},
        {
            "kind": PhaseKind.PROGRESSIVE_RATIO,
            "steps": 5,
            "pr_start": 1,
            "pr_increment": 1,
            "ratio": 2,
        },
        {"kind": PhaseKind.EXTINCTION, "steps": 5, "ratio": 2},
        {"kind": PhaseKind.EXTINCTION, "steps": 5, "pr_start": 1},
        {"kind": PhaseKind.EXTINCTION, "steps": 5, "pr_increment": 0},
        {"kind": PhaseKind.EXTINCTION, "steps": 0},
        {"kind": PhaseKind.FIXED_RATIO, "steps": 0, "ratio": 1},
    ],
)
def test_phase_spec_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        PhaseSpec(**kwargs)


def test_phase_spec_accepts_valid() -> None:
    fr(1, 1)
    pr(1, 0, 1)
    ext(1)


def test_world_config_rejects_empty_schedule_and_nonpositive_reward() -> None:
    with pytest.raises(ValidationError):
        WorldConfig(active_lever="press_a", reward_magnitude=1.0, schedule=[])
    with pytest.raises(ValidationError):
        WorldConfig(active_lever="press_a", reward_magnitude=0.0, schedule=[ext(1)])
