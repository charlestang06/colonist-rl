"""Small command-line entry points; all training artifacts stay local."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parser():
    from .native_opponents import NATIVE_IDS

    root = argparse.ArgumentParser(description="Local Catan simulation and recurrent PPO")
    commands = root.add_subparsers(dest="command", required=True)
    board = commands.add_parser("board", help="inspect variable board specifications")
    board.add_argument("--players", type=int, choices=(4, 6), default=4)
    board.add_argument("--seed", type=int, default=0)
    capture = commands.add_parser("capture", help="read a verified annotated screenshot offline")
    capture.add_argument("--annotation", type=Path, required=True)
    capture.add_argument("--screenshot", type=Path, required=True)
    capture.add_argument("--output", type=Path)
    export = commands.add_parser("export", help="export a simulator player observation as JSON")
    export.add_argument("--seed", type=int, default=0)
    export.add_argument("--output", type=Path, required=True)
    train = commands.add_parser(
        "train", help="train PPO, optionally imitating heuristic moves first"
    )
    train.add_argument("--updates", type=_positive, default=10)
    train.add_argument("--rollout", type=_positive, default=64)
    train.add_argument("--epochs", type=_positive, default=4)
    train.add_argument("--hidden-size", type=_positive, default=64)
    train.add_argument("--warmstart-decisions", type=int, default=256)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    train.add_argument("--resume", type=Path)
    train.add_argument("--opponent-checkpoint", type=Path, action="append", default=[])
    train.add_argument(
        "--historical-opponents",
        action="store_true",
        help="add frozen snapshots to the opponent pool after each update",
    )
    train.add_argument("--max-game-steps", type=_positive, default=20000)
    for name in ("simulate", "evaluate"):
        command = commands.add_parser(name, help="run seat-rotated held-out games")
        command.add_argument("--games", type=_positive, default=16)
        command.add_argument("--seed", type=int, default=100000)
        command.add_argument(
            "--baseline",
            choices=("random", "heuristic", "greedy_no_trade") + NATIVE_IDS,
            default="heuristic",
        )
        command.add_argument(
            "--candidate", choices=("heuristic", "random", "greedy_no_trade"), default="heuristic"
        )
        command.add_argument(
            "--opponent-lineup",
            nargs=3,
            choices=("random", "heuristic", "greedy_no_trade") + NATIVE_IDS,
        )
        command.add_argument(
            "--wall-seconds",
            type=float,
            help="whole-run budget; native opponents require this and an external worker hard cap",
        )
        command.add_argument("--max-game-steps", type=_positive, default=20000)
        command.add_argument("--output", type=Path)
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path)
            command.add_argument("--compare-rejection", action="store_true")
    experiment = commands.add_parser("experiment", help="run a bounded reproducible learning pilot")
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--decisions", type=_positive, default=2500)
    experiment.add_argument("--epochs", type=_positive, default=2)
    experiment.add_argument("--hidden-size", type=_positive, default=32)
    experiment.add_argument("--wall-seconds", type=float, default=900)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "experiment":
        from .experiments import run_experiment

        result = run_experiment(
            args.output,
            decisions=args.decisions,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            wall_seconds=args.wall_seconds,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    if args.command in ("capture", "export"):
        from .perception import observation_to_dict, read_capture

        if args.command == "capture":
            view = read_capture(args.annotation, args.screenshot).observation
        else:
            from .simulator import FourPlayerGame

            view = FourPlayerGame().reset(args.seed)
        rendered = json.dumps(observation_to_dict(view), indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        else:
            print(rendered)
        return 0
    if args.command == "board":
        from .board import BoardSpec, RulesConfig

        board = (
            BoardSpec.standard(args.seed) if args.players == 4 else BoardSpec.extended(args.seed)
        )
        rules = RulesConfig.standard() if args.players == 4 else RulesConfig.extended()
        print(
            json.dumps(
                {
                    "players": args.players,
                    "tiles": len(board.tiles),
                    "vertices": len(board.vertices),
                    "edges": len(board.edges),
                    "ports": len(board.ports),
                    "special_build": rules.special_build,
                    "simulation_available": args.players == 4,
                },
                indent=2,
            )
        )
        return 0
    import torch

    from .training import (
        PPOConfig,
        RolloutCollector,
        evaluate,
        imitation_warmstart,
        load_checkpoint,
        ppo_update,
        save_checkpoint,
    )

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.command == "train":
        from .policy import GraphPolicy

        if args.warmstart_decisions < 0:
            raise ValueError("warmstart decisions cannot be negative")
        state = None
        if args.resume:
            policy, state = load_checkpoint(args.resume)
        else:
            policy = GraphPolicy(args.hidden_size)
            if args.warmstart_decisions:
                print(
                    json.dumps(
                        {
                            "warmstart": imitation_warmstart(
                                policy, decisions=args.warmstart_decisions, seed=args.seed
                            )
                        }
                    ),
                    flush=True,
                )
        config = PPOConfig(epochs=args.epochs)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        if state and state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        steps = state["steps"] if state else 0
        rng_before_history = torch.get_rng_state(), random.getstate()
        history = [
            load_checkpoint(path)[0].eval().requires_grad_(False)
            for path in args.opponent_checkpoint
        ]
        torch.set_rng_state(rng_before_history[0])
        random.setstate(rng_before_history[1])
        collector = RolloutCollector(
            policy,
            seed=args.seed + steps,
            max_game_steps=args.max_game_steps,
            historical_policies=tuple(history),
        )
        start = time.monotonic()
        for update in range(args.updates):
            rollout = collector.collect(args.rollout)
            metrics = ppo_update(policy, optimizer, rollout, config)
            steps += len(rollout)
            if args.historical_opponents:
                history.append(copy.deepcopy(policy).eval().requires_grad_(False))
                history = history[-3:]
                collector.historical = tuple(history)
            metadata = {
                "seed": args.seed,
                "discount": 1.0,
                "vp_shaping_scale": 0.2,
                "warmstart_decisions": 0 if args.resume else args.warmstart_decisions,
                "incoming_trades_only": True,
                "resume_starts_fresh_match": True,
            }
            save_checkpoint(args.checkpoint, policy, optimizer, steps=steps, metadata=metadata)
            print(
                json.dumps(
                    {
                        "update": update + 1,
                        "learner_decisions": steps,
                        "episodes": collector.episodes,
                        "truncated": collector.truncated,
                        "trade_responses": collector.trade_responses,
                        "seconds": time.monotonic() - start,
                        **metrics,
                    }
                ),
                flush=True,
            )
        return 0
    policy = None
    if args.command == "evaluate" and args.checkpoint:
        policy = load_checkpoint(args.checkpoint)[0].eval()
    torch.manual_seed(args.seed)
    kwargs = {
        "games": args.games,
        "seed": args.seed,
        "baseline": args.baseline,
        "max_game_steps": args.max_game_steps,
        "candidate": args.candidate,
        "opponent_lineup": args.opponent_lineup,
        "deadline": time.monotonic() + args.wall_seconds if args.wall_seconds is not None else None,
    }
    result = evaluate(policy, **kwargs)
    if args.command == "evaluate" and args.compare_rejection:
        if policy is None:
            raise ValueError("--compare-rejection requires --checkpoint")
        torch.manual_seed(args.seed)
        result = {
            "learned_response": result,
            "always_reject": evaluate(policy, reject_trades=True, **kwargs),
        }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0
