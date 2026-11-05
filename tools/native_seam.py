"""Engineering-only native gateway seam; root-scheduled external v4 guard required.

Preparation must not execute this file. No training, labels or full games. At most
8 authentic decide entries, including direct parity calls. Test hooks never change
installed sources or native root menus and are named separately in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import signal
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


class BudgetExpired(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--fixture-seed', type=int, required=True)
    parser.add_argument('--budget-seconds', type=float, default=12)
    parser.add_argument('--authorized-root-go', action='store_true', required=True)
    args = parser.parse_args()
    if not 0 < args.budget_seconds <= 27:
        parser.error('internal budget must be >0 and <=27; external hard cap remains <=30s')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.budget_seconds
    manifest = {
        'schema': 'colonist.native-engineering-seam/1', 'status': 'running',
        'namespace': args.namespace, 'fixture_seed': args.fixture_seed,
        'purpose': 'engineering_seam_only_not_playing_strength',
        'information_regime': 'public_only_candidate_vs_privileged_native_opponents',
        'adapter_suffix': '+atomic_discard_reject_trade_v1', 'full_rule_equivalence': False,
        'limits': {'internal_seconds': args.budget_seconds, 'external_hard_seconds_max': 30,
                   'external_rss_bytes_max': 2147483648, 'cpu_threads': 1,
                   'max_authentic_decide_entries': 8},
        'external_cap_status': 'caller_must_join_v4_guard_finish_and_resource_records',
        'authentic_query_entries': 0, 'authentic_query_completions': 0,
        'queries': [], 'cases': [], 'test_hooks': [],
        'planned_cases': [
            'real_setup_color_and_coordinate_mapping',
            'normal_action_conversion_and_execution_matrix',
            'parity_seat0_vf_initial_nonforced',
            'parity_seat1_ab_sparse_normal_nonforced',
            'parity_seat2_vf_forced_roll',
            'parity_seat3_ab_forced_end',
            'real_backend_exception_isolation_stub',
            'real_rolled_seven_discards_cache_renewal_and_synthetic_trade',
            'native82_preserved_chance_mismatch_characterization',
            'deadline_incomplete_native_metadata_without_overrun_step',
        ],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': sys.executable, 'source_root': str(Path(__file__).resolve().parents[1]),
    }

    def save():
        manifest['elapsed_seconds'] = time.monotonic() - started
        passed = {row['name'] for row in manifest['cases'] if row['status'] == 'passed'}
        manifest['untested_or_unpassed_cases'] = [name for name in manifest['planned_cases'] if name not in passed]
        tmp = args.output / 'manifest.json.tmp'
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
        tmp.replace(args.output / 'manifest.json')

    def check():
        if time.monotonic() >= deadline:
            raise BudgetExpired('internal seam budget exhausted; remaining cases untested')

    def interrupted(signum, frame):
        raise BudgetExpired(f'external resource/deadline signal {signum}')

    for sig in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM):
        signal.signal(sig, interrupted)

    def case(name, fn):
        check()
        row = {'name': name, 'status': 'running', 'authentic_queries_before': manifest['authentic_query_entries']}
        manifest['cases'].append(row)
        save()
        before = time.perf_counter()
        try:
            row['assertions'] = fn() or {}
            check()
            row['status'] = 'passed'
        except BaseException:
            row['status'] = 'incomplete' if time.monotonic() >= deadline else 'failed_or_interrupted'
            raise
        finally:
            row['seconds'] = time.perf_counter() - before
            row['authentic_query_entries'] = manifest['authentic_query_entries'] - row['authentic_queries_before']
            save()

    def query(label, fn, game, actions):
        check()
        if manifest['authentic_query_entries'] >= 8:
            raise BudgetExpired('eight authentic decide entries already used')
        record = {'label': label, 'status': 'entered', 'root_count': len(actions)}
        manifest['queries'].append(record)
        manifest['authentic_query_entries'] += 1
        save()
        before = time.perf_counter()
        try:
            answer = fn(game, actions)
            record['status'] = 'returned'
            manifest['authentic_query_completions'] += 1
            check()
            return answer
        finally:
            record['seconds'] = time.perf_counter() - before
            save()

    save()
    try:
        # Imported only in the authorized worker, never by static preparation.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            if os.environ.get(key) != '1':
                raise RuntimeError(f'{key}=1 required by root-scheduled guard protocol')
        from catanatron.models.enums import Action as NativeAction
        from catanatron.models.enums import ActionPrompt, ActionRecord, ActionType
        from catanatron.players.tree_search_utils import execute_spectrum

        from colonist_rl import evaluation
        from colonist_rl.contracts import RESOURCES, Action, TradeOffer
        from colonist_rl.native_opponents import (
            NATIVE_IDS,
            NativeOpponentRunner,
            _native_player_class,
            verify_native_installation,
        )
        from colonist_rl.simulator import _RNG_LOCK, FourPlayerGame
        from colonist_rl.training import _lineup, _MatchDriver
        check()
        manifest['engine'] = verify_native_installation()
        manifest['torch_threads'] = torch.get_num_threads()
        manifest['thread_environment'] = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
        save()

        def snapshot(env):
            # Hash only: no native private state, RNG or action records exported.
            value = (env.game.__dict__, env.version, env._rng_state,
                     env._pending, env._discard_remaining)
            return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()

        def grant(env, seat, hand):
            before = [env._hand(p) for p in range(4)]
            for i, resource in enumerate(RESOURCES):
                delta = hand[i] - before[seat][i]
                env.game.state.player_state[f'P{seat}_{resource}_IN_HAND'] += delta
                env.game.state.resource_freqdeck[i] -= delta
            env._record_known_changes(before, 'ENGINEERING_FIXTURE_PUBLIC_GRANT', seat)
            env._refresh_actions()
            env.assert_resource_conservation()

        def main_turn(env, seat, *, rolled=True):
            state = env.game.state
            assert not state.is_initial_build_phase
            state.current_turn_index = state.current_player_index = seat
            state.current_prompt = ActionPrompt.PLAY_TURN
            state.is_discarding = state.is_moving_knight = state.is_road_building = False
            state.free_roads_available = 0
            env._discard_remaining = {}
            for p in range(4):
                state.player_state[f'P{p}_HAS_ROLLED'] = bool(rolled and p == seat)
                state.player_state[f'P{p}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'] = False
            env._refresh_actions()

        def normalized(env):
            for p in range(4):
                grant(env, p, (0, 0, 0, 0, 0))
            return env

        def give_dev(env, seat, card):
            env.game.state.development_listdeck.remove(card)
            env.game.state.player_state[f'P{seat}_{card}_IN_HAND'] += 1
            env.game.state.player_state[f'P{seat}_{card}_OWNED_AT_START'] = True
            env._refresh_actions()

        def members(env):
            view = env.observe()
            converted = []
            for action in env.game.playable_actions:
                mapped = env._convert(action)
                assert mapped in view.legal_actions
                kind, value = action.action_type.name, action.value
                if kind in ('BUILD_SETTLEMENT', 'BUILD_CITY'):
                    assert env._vertex_to_node[mapped.target] == value
                elif kind == 'BUILD_ROAD':
                    assert tuple(sorted(env._id_to_edge[mapped.target])) == tuple(sorted(value))
                elif kind == 'MOVE_ROBBER':
                    assert env._tile_to_coordinate[mapped.target] == value[0]
                    assert mapped.partner == (None if value[1] is None else env.game.state.color_to_index[value[1]])
                elif kind == 'MARITIME_TRADE':
                    assert mapped.give == tuple(value[:-1].count(r) for r in RESOURCES)
                    assert mapped.receive == tuple(int(value[-1] == r) for r in RESOURCES)
                elif kind == 'PLAY_YEAR_OF_PLENTY':
                    assert (mapped.resource, mapped.second_resource) == (value[0], value[1] if len(value) > 1 else None)
                elif kind == 'PLAY_MONOPOLY':
                    assert mapped.resource == value
                converted.append(kind)
            return set(converted)

        base = FourPlayerGame(max_steps=1000)
        base.reset(args.fixture_seed)
        setups = []
        covered = set()

        def setup_case():
            for seat in range(4):
                for identity in NATIVE_IDS:
                    runner = NativeOpponentRunner(base, seat, identity, planning_seed=71 + seat)
                    assert runner._player.color == base.game.state.colors[seat]
            while base.game.state.is_initial_build_phase:
                check()
                assert len(setups) < 20
                covered.update(members(base))
                setups.append(deepcopy(base))
                action = base.observe().legal_actions[0]
                old = len(base.observe().buildings), len(base.observe().roads)
                base.step(action)
                now = len(base.observe().buildings), len(base.observe().roads)
                assert now == (old[0] + 1, old[1]) if action.kind == 'BUILD_SETTLEMENT' else now == (old[0], old[1] + 1)
                base.assert_resource_conservation()
            normalized(base)
            return {'all_seat_colors': True, 'seat_colors': [c.name for c in base.game.state.colors], 'initial_steps': len(setups), 'kinds': sorted(covered)}
        case('real_setup_color_and_coordinate_mapping', setup_case)
        manifest['test_hooks'].append('engineering_fixture_hand/dev/turn_preparation; no native root masking')

        def conversion_case():
            rich = deepcopy(base)
            main_turn(rich, 0)
            grant(rich, 0, (4, 4, 4, 4, 4))
            for p in (1, 2, 3):
                grant(rich, p, (1, 0, 0, 0, 0))
            for card in ('KNIGHT', 'MONOPOLY', 'YEAR_OF_PLENTY', 'ROAD_BUILDING'):
                give_dev(rich, 0, card)
            covered.update(members(rich))
            kinds = ('BUILD_CITY', 'BUILD_ROAD', 'MARITIME_TRADE', 'BUY_DEVELOPMENT_CARD',
                     'PLAY_KNIGHT_CARD', 'PLAY_YEAR_OF_PLENTY', 'PLAY_MONOPOLY', 'PLAY_ROAD_BUILDING', 'END_TURN')
            for kind in kinds:
                check()
                env = deepcopy(rich)
                a = next(a for a in env.observe().legal_actions if a.kind == kind)
                before = env._hand(0)
                deck_size = len(env.game.state.development_listdeck)
                road_count = len(env.observe().roads)
                env.step(a)
                env.assert_resource_conservation()
                if kind == 'BUILD_CITY':
                    assert any(b.vertex == a.target and b.level == 2 for b in env.observe(0).buildings)
                elif kind == 'BUILD_ROAD':
                    assert len(env.observe().roads) == road_count + 1
                elif kind == 'MARITIME_TRADE':
                    assert env._hand(0) == tuple(h - g + r for h, g, r in zip(before, a.give, a.receive))
                elif kind == 'BUY_DEVELOPMENT_CARD':
                    assert len(env.game.state.development_listdeck) == deck_size - 1
                elif kind == 'PLAY_KNIGHT_CARD':
                    assert env.observe().context.kind == 'MOVE_ROBBER'
                elif kind == 'PLAY_ROAD_BUILDING':
                    assert env.observe().context.free_roads == 2
                elif kind == 'PLAY_YEAR_OF_PLENTY':
                    chosen = (a.resource, a.second_resource)
                    assert env._hand(0) == tuple(h + chosen.count(r) for h, r in zip(before, RESOURCES))
                elif kind == 'PLAY_MONOPOLY':
                    assert all(env._hand(p)[RESOURCES.index(a.resource)] == 0 for p in (1, 2, 3))
                elif kind == 'END_TURN':
                    assert env.game.state.current_turn_index == 1
            robber = deepcopy(rich)
            robber.game.state.current_prompt = ActionPrompt.MOVE_ROBBER
            robber.game.state.is_moving_knight = True
            robber._refresh_actions()
            covered.update(members(robber))
            a = next(a for a in robber.observe().legal_actions if a.partner is not None)
            victim, own = sum(robber._hand(a.partner)), sum(robber._hand(0))
            robber.step(a)
            assert sum(robber._hand(a.partner)) == victim - 1 and sum(robber._hand(0)) == own + 1
            roll = deepcopy(base)
            main_turn(roll, 2, rolled=False)
            covered.update(members(roll))
            with patch('catanatron.apply_action.roll_dice', return_value=(2, 3)):
                roll.step(Action('ROLL'))
            assert roll.observe().context.has_rolled
            required = {'BUILD_SETTLEMENT', 'BUILD_CITY', 'BUILD_ROAD', 'MOVE_ROBBER', 'MARITIME_TRADE', 'PLAY_YEAR_OF_PLENTY', 'PLAY_MONOPOLY', 'BUY_DEVELOPMENT_CARD', 'PLAY_KNIGHT_CARD', 'PLAY_ROAD_BUILDING', 'ROLL', 'END_TURN'}
            assert required <= covered
            return {'mapped_kinds': sorted(covered), 'real_wrapper_steps_and_conservation': True}
        manifest['test_hooks'].append('conversion-only forced roll_dice=(2,3)')
        case('normal_action_conversion_and_execution_matrix', conversion_case)

        def parity(env, seat, identity, label, nonforced=False):
            check()
            seed = args.fixture_seed * 97 + seat
            runner = NativeOpponentRunner(env, seat, identity, planning_seed=seed)
            root = deepcopy(env.game.playable_actions)
            assert env.current_player == seat
            assert not nonforced or len(root) > 1
            captured = []
            original = runner._player.decide
            def tracked(game, actions):
                assert list(actions) == root and list(game.playable_actions) == root
                answer = query(label + '/runner', original, game, actions)
                captured.append(answer)
                return answer
            runner._player.decide = tracked
            before, external = snapshot(env), random.getstate()
            start_rng = runner._rng_state
            mapped = runner.select_action()
            assert snapshot(env) == before and random.getstate() == external
            assert any(mapped is a for a in env.observe().legal_actions) or mapped in env.observe().legal_actions
            end_rng = runner._rng_state
            direct = _native_player_class(identity)(env.game.state.colors[seat], **runner._configuration)
            with _RNG_LOCK:
                outer = random.getstate()
                random.setstate(start_rng)
                try:
                    copied = deepcopy(env.game)
                    copied.playable_actions = deepcopy(root)
                    answer = query(label + '/direct', direct.decide, copied, list(copied.playable_actions))
                    direct_end_rng = random.getstate()
                finally:
                    random.setstate(outer)
            assert snapshot(env) == before and random.getstate() == external
            assert answer == captured[0] and env._convert(answer) == mapped
            assert end_rng == direct_end_rng
            env.step(mapped)
            env.assert_resource_conservation()
            return {'seat': seat, 'identity': identity, 'nonforced': len(root) > 1,
                    'native_root_count': len(root), 'exact_return_parity': True,
                    'true_state_rng_and_global_unchanged': True,
                    'planning_rng_advanced': end_rng != start_rng,
                    'original_root_order_equal': True, 'mapped_kind': mapped.kind,
                    'runner_metadata': runner.metadata()}

        # Exactly four pairs / eight native decisions if the guard permits.
        case('parity_seat0_vf_initial_nonforced', lambda: parity(deepcopy(setups[0]), 0, NATIVE_IDS[0], 'seat0-vf-setup', True))
        sparse = deepcopy(base)
        main_turn(sparse, 1)
        grant(sparse, 1, (1, 1, 0, 0, 0))
        assert {a.action_type.name for a in sparse.game.playable_actions} <= {'END_TURN', 'BUILD_ROAD'}
        case('parity_seat1_ab_sparse_normal_nonforced', lambda: parity(sparse, 1, NATIVE_IDS[1], 'seat1-ab-sparse', True))
        forced = deepcopy(base)
        main_turn(forced, 2, rolled=False)
        assert [a.action_type for a in forced.game.playable_actions] == [ActionType.ROLL]
        case('parity_seat2_vf_forced_roll', lambda: parity(forced, 2, NATIVE_IDS[0], 'seat2-vf-roll'))
        ended = deepcopy(base)
        main_turn(ended, 3)
        assert [a.action_type for a in ended.game.playable_actions] == [ActionType.END_TURN]
        case('parity_seat3_ab_forced_end', lambda: parity(ended, 3, NATIVE_IDS[1], 'seat3-ab-end'))

        def isolation_exception():
            env = deepcopy(base)
            main_turn(env, 1)
            runner = NativeOpponentRunner(env, 1, NATIVE_IDS[0], planning_seed=9)
            env.game.state.action_records.append(ActionRecord(
                NativeAction(env.game.state.colors[1], ActionType.END_TURN, None),
                ['engineering-nested-result']))
            before, outer, planning = snapshot(env), random.getstate(), runner._rng_state
            def stub(game, actions):
                random.random()
                game.state.development_listdeck.reverse()
                game.state.player_state['P0_WOOD_IN_HAND'] += 1
                game.playable_actions.clear()
                game.state.action_records[-1].result.append('controlled-copy-mutation')
                game.state.action_records.clear()
                raise RuntimeError('controlled isolation stub')
            runner._player.decide = stub
            try:
                runner.select_action()
            except RuntimeError as exc:
                assert str(exc) == 'controlled isolation stub'
            else:
                raise AssertionError('controlled stub did not raise')
            assert before == snapshot(env) and random.getstate() == outer
            assert runner._rng_state != planning
            return {'stub_only_no_authentic_query': True, 'real_backend_immutable': True, 'independent_rng_advanced_external_restored': True}
        manifest['test_hooks'].append('exception isolation: synthetic nested record plus instance decide stub mutates real copied backend and consumes planning random')
        case('real_backend_exception_isolation_stub', isolation_exception)

        def discard_and_trade():
            env = deepcopy(base)
            main_turn(env, 2, rolled=False)
            grant(env, 0, (3, 2, 1, 2, 2))
            grant(env, 1, (2, 2, 2, 2, 2))
            with patch('catanatron.apply_action.roll_dice', return_value=(3, 4)):
                env.step(Action('ROLL'))
            assert env.current_player == 0 and env.game.state.current_turn_index == 2
            runners = {p: NativeOpponentRunner(env, p, NATIVE_IDS[p % 2], planning_seed=90+p) for p in (0, 1)}
            counts = {}
            def finish():
                while env.observe().context.kind == 'DISCARD':
                    check()
                    view = env.observe()
                    actor, remaining, version = view.player_id, view.context.discard_remaining, view.context.version
                    before, outer = snapshot(env), random.getstate()
                    action = runners[actor].select_action()
                    assert snapshot(env) == before and random.getstate() == outer
                    assert action.kind == 'DISCARD_RESOURCE' and view.hand[RESOURCES.index(action.resource)] > 0
                    env.step(action)
                    counts[actor] = counts.get(actor, 0) + 1
                    assert env.version == version + 1
                    if remaining > 1:
                        assert env.current_player == actor and env.observe().context.discard_remaining == remaining - 1
                    env.assert_resource_conservation()
            finish()
            assert counts == {0: 5, 1: 5} and env.current_player == 2
            assert env.observe().context.kind == 'MOVE_ROBBER'
            first_cache = runners[0]._discard
            main_turn(env, 2, rolled=False)
            grant(env, 0, (3, 2, 1, 2, 2))
            with patch('catanatron.apply_action.roll_dice', return_value=(3, 4)):
                env.step(Action('ROLL'))
            finish()
            assert counts == {0: 10, 1: 5} and runners[0]._discard != first_cache
            assert all(r.metrics()['native_decisions'] == 0 for r in runners.values())
            assert runners[0].metrics()['atomic_discard_shim_calls'] == 10
            trade = deepcopy(base)
            main_turn(trade, 0)
            grant(trade, 0, (1, 0, 0, 0, 0))
            grant(trade, 1, (0, 1, 0, 0, 0))
            before_hands = [trade._hand(p) for p in range(4)]
            trade.offer_trade(TradeOffer('engineering-reject', 0, 1, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
            runner = NativeOpponentRunner(trade, 1, NATIVE_IDS[1], planning_seed=71)
            action = runner.select_action()
            assert action.kind == 'REJECT_TRADE'
            trade.step(action)
            assert [trade._hand(p) for p in range(4)] == before_hands and trade._pending is None
            main_turn(trade, 1)
            driver = _MatchDriver(trade, 0, args.fixture_seed, baseline=NATIVE_IDS[0], allow_native=True)
            before = snapshot(trade)
            driver.maybe_offer()
            assert snapshot(trade) == before and driver.offers == 0
            return {'opponent_out_of_turn_discard': True, 'cards_discarded': sum(counts.values()), 'cache_renewal': True, 'shim_native_decisions': 0, 'trade_reject_and_no_proposal': True}
        manifest['test_hooks'].append('actual wrapper rolled-seven fixtures: roll_dice=(3,4), repeated phase with own-hand grant')
        case('real_rolled_seven_discards_cache_renewal_and_synthetic_trade', discard_and_trade)

        def chance_characterization():
            roll = deepcopy(base)
            main_turn(roll, 0, rolled=False)
            action = NativeAction(roll.game.state.colors[0], ActionType.ROLL, (1, 1))
            with patch('catanatron.apply_action.roll_dice', return_value=(2, 3)):
                actual = deepcopy(roll.game).execute(action, validate_action=False)
                assert actual.result == (2, 3)
                replay = deepcopy(roll.game).execute(action, validate_action=False, action_record=ActionRecord(action, (6, 6)))
                assert replay.result == (6, 6)
                branches = execute_spectrum(deepcopy(roll.game), NativeAction(action.color, ActionType.ROLL, None))
                assert len(branches) == 11 and all(g.state.action_records[-1].result == (2, 3) for g, _ in branches)
            buy = deepcopy(base)
            main_turn(buy, 0)
            grant(buy, 0, (0, 0, 1, 1, 1))
            deck = buy.game.state.development_listdeck
            index = deck.index('KNIGHT')
            deck[-1], deck[index] = deck[index], deck[-1]
            action = NativeAction(buy.game.state.colors[0], ActionType.BUY_DEVELOPMENT_CARD, 'MONOPOLY')
            actual = deepcopy(buy.game).execute(action, validate_action=False)
            assert actual.result == 'KNIGHT'
            replay = deepcopy(buy.game).execute(action, validate_action=False, action_record=ActionRecord(action, 'MONOPOLY'))
            assert replay.result == 'MONOPOLY'
            branches = execute_spectrum(deepcopy(buy.game), action)
            assert len(branches) > 1 and all(g.state.action_records[-1].result == 'KNIGHT' for g, _ in branches)
            return {'roll_action_value_ignored_without_record': True, 'roll_record_result_honored': True, 'all_11_roll_branches_follow_hook': True, 'development_value_ignored_actual_top_drawn': True, 'development_record_result_honored': True, 'native_source_modified': False}
        manifest['test_hooks'].append('chance characterization only: fixed roll RNG hook, fixture deck top KNIGHT, explicit ActionRecord replay; no baseline algorithm patch')
        case('native82_preserved_chance_mismatch_characterization', chance_characterization)

        def deadline_plumbing():
            prepared = deepcopy(base)
            main_turn(prepared, 1)
            original_version = prepared.version
            clock = [0.0]
            def factory(**kwargs):
                env = deepcopy(prepared)
                env.reset = lambda seed: env.observe()
                env.step = lambda action: (_ for _ in ()).throw(AssertionError('overrun step must not execute'))
                return env
            def expiring(runner):
                view = runner._env.observe()
                runner._counts['deadline_plumbing_stub_calls'] += 1
                runner._phase_counts[view.context.kind] += 1
                runner._phase_seconds[view.context.kind] += 0.001
                clock[0] = 2.0
                return view.legal_actions[0]
            with patch.object(evaluation, 'FourPlayerGame', side_effect=factory), patch.object(evaluation.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(NativeOpponentRunner, 'select_action', expiring):
                result = evaluation.evaluate_blocks(board_seeds=[args.fixture_seed], baseline=NATIVE_IDS[0], deadline=1.0, max_game_steps=1000)
            assert result['status'] == 'incomplete' and result['incomplete_reason'] == 'deadline'
            row = result['interrupted_game']
            assert row['steps'] == original_version and len(row['native_opponents']) == 3
            assert sum(x['metrics'].get('deadline_plumbing_stub_calls', 0) for x in row['native_opponents']) == 1
            assert sum(x['metrics']['total_planning_seconds'] for x in row['native_opponents']) >= 0.001
            assert all(x['metadata']['information_regime'].startswith('public_only') for x in row['native_opponents'])
            for identity in NATIVE_IDS:
                try:
                    evaluation.evaluate_blocks(board_seeds=[args.fixture_seed], candidate=identity)
                except ValueError:
                    pass
                else:
                    raise AssertionError('native candidate accepted')
                try:
                    _lineup((identity, 'random', 'heuristic'))
                except ValueError:
                    pass
                else:
                    raise AssertionError('native training lineup accepted')
            return {'stub_only_no_authentic_query': True, 'incomplete_with_native_metadata_latency': True, 'overrun_action_not_applied': True, 'candidate_training_rejection': True}
        manifest['test_hooks'].append('deadline plumbing only: existing real prepared fixture with reset reuse, native select_action timing stub and fake monotonic clock')
        case('deadline_incomplete_native_metadata_without_overrun_step', deadline_plumbing)
        assert manifest['authentic_query_entries'] <= 8
        manifest['engine_after_hooks'] = verify_native_installation()
        manifest['status'] = 'passed_engineering_seams_only'
        manifest['untested'] = ['full_games', 'playing_strength', 'equal_information_baselines', 'actual_AB_timeout_cost']
        save()
        return 0
    except BudgetExpired as exc:
        manifest['status'] = 'incomplete_censored'
        manifest['reason'] = str(exc)
        manifest['untested'] = 'all cases not marked passed; no automatic retry or budget escalation'
        save()
        return 124
    except Exception as exc:  # noqa: BLE001 - persist failure evidence, then exit nonzero
        manifest['status'] = 'failed'
        manifest['failure_type'] = type(exc).__name__
        manifest['failure'] = str(exc)
        manifest['traceback'] = traceback.format_exc()
        save()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
