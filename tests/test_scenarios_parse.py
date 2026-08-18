"""Every scenario in run_scenarios.sh must survive train.py's argparse.

WHY THIS EXISTS
---------------
`literature_baseline_diff_l1_organ_groupnorm_adv/` contains a run_scenarios.log
and nothing else. Its last line is:

    train.py: error: unrecognized arguments: --use_adversarial --use_cond_disc
                                             --adv_warmup_epochs 15

The scenario was written against flags that train.py did not have yet, argparse
rejected the invocation, and the row produced no model. `run_one` does surface a
non-zero exit, so the failure was in the log the whole time and was simply never
read — and DIFFUSION_PLAN.md section 11 calls that particular row the comparison
twin without which an adversarial diffusion run "says nothing". Three adversarial
diffusion runs were launched anyway.

The check is cheap and catches the whole class: parse each scenario's flags with
the real parser and assert it does not exit. No data, no GPU, no checkpoints —
it drives `train._parse` by swapping sys.argv and stops there.

RUN IT BEFORE EVERY LAUNCH:
    python -m pytest tests/test_scenarios_parse.py -q
    python tests/test_scenarios_parse.py          # standalone, same checks
"""
import pathlib
import re
import shlex
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# pytest is optional. conftest.py records that most tests in this repo are
# `__main__` scripts run directly, and the training host is not guaranteed to
# have pytest installed — a launch gate that cannot run without a test framework
# is a gate that gets skipped. The `__main__` block below performs every check.
try:
    import pytest
except ImportError:                                  # pragma: no cover
    class _NoPytest:
        """Minimal stand-in so the module still imports and runs standalone."""
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)

        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn
    pytest = _NoPytest()

SCENARIOS_SH = _ROOT / 'run_scenarios.sh'

# "name|--flag1 --flag2 ...", one per line inside the SCENARIOS=( ... ) array.
# Commented-out lines are skipped: a leading '#' means the row is deliberately
# parked, and parking a row should not require it to stay flag-valid.
_ROW = re.compile(r'^\s*"([^"|]+)\|([^"]*)"\s*$')


def _scenarios():
    """[(name, flags_str)] read straight out of the shell array."""
    rows, in_array = [], False
    for line in SCENARIOS_SH.read_text().splitlines():
        if line.startswith('SCENARIOS=('):
            in_array = True
            continue
        if in_array and line.startswith(')'):
            break
        if not in_array or line.lstrip().startswith('#'):
            continue
        m = _ROW.match(line)
        if m:
            rows.append((m.group(1), m.group(2)))
    return rows


def _parse_flags(flags: str):
    """Run train.py's real parser over one scenario's flags.

    Drives `train._parse`, which is the actual parser the real invocation uses —
    checking against a reconstructed copy would let the two drift, which is the
    exact failure mode this gate exists to catch.

    `--output_dir` and `--seed` are prepended because run_one always supplies
    them; without them a required-argument error would be attributed to the
    scenario rather than to this harness. Returns the parsed namespace or raises
    SystemExit exactly as the real invocation would.
    """
    import train

    argv = ['train.py', '--output_dir', '/tmp/_parse_check', '--seed', '42']
    argv += shlex.split(flags)
    old, sys.argv = sys.argv, argv
    try:
        return train._parse()
    finally:
        sys.argv = old


def test_scenarios_file_is_readable():
    assert SCENARIOS_SH.is_file(), f'missing {SCENARIOS_SH}'
    assert _scenarios(), 'no scenarios parsed out of run_scenarios.sh — the ' \
                         'SCENARIOS=( ... ) format changed and this gate is blind'


@pytest.mark.parametrize('name,flags', _scenarios(), ids=lambda v: v if isinstance(v, str) else '')
def test_scenario_flags_parse(name, flags):
    """The gate. A SystemExit here is argparse rejecting the row."""
    try:
        _parse_flags(flags)
    except SystemExit as e:
        pytest.fail(
            f"scenario '{name}' does not parse against train.py's argparse "
            f"(exit {e.code}). This is the failure that silently cost "
            f"diff_l1_organ_groupnorm_adv its entire run. Flags:\n  {flags}")


def test_scenario_names_are_unique():
    """Duplicate names make `run_scenarios.sh <name>` run two different configs
    into the same output dir, and the second silently overwrites the first."""
    names = [n for n, _ in _scenarios()]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f'duplicate scenario names: {sorted(dupes)}'


def test_no_retired_lambda_adv():
    """lambda_adv >= 2.0 is retired on the diffusion path.

    At 2.0, samples/ep080.png of diff_v_organ_adv carries a regular diagonal
    hatching pattern over every output including pure air, with PSNR collapsed to
    19-22 against 28-30 for the same config with no critic. See config.py's
    LAMBDA_ADV note. This asserts the value cannot creep back in via a scenario.
    """
    bad = []
    for name, flags in _scenarios():
        if '--use_diffusion' not in flags:
            continue
        toks = shlex.split(flags)
        if '--lambda_adv' in toks:
            v = float(toks[toks.index('--lambda_adv') + 1])
            if v >= 2.0:
                bad.append((name, v))
        elif '--use_adversarial' in toks:
            # relying on the config default is fine only while that default is <2
            import config
            if config.LAMBDA_ADV >= 2.0:
                bad.append((name, config.LAMBDA_ADV))
    assert not bad, (f'lambda_adv >= 2.0 on the diffusion path: {bad}. '
                     f'That value produced the diagonal-hatching artifact.')


def test_adversarial_diffusion_rows_carry_stabilisers():
    """An adversarial diffusion run with default stabilisers is a known dead end.

    Every such run so far sat at train_disc ~0.02 / train_adv ~0.95 past epoch 80
    — D(real)~0.81, D(fake)~0.03, i.e. the critic had won outright and G's
    adversarial gradient was noise for the whole run. lambda does not fix it.
    Require at least the two that address it directly.
    """
    missing = []
    for name, flags in _scenarios():
        if '--use_diffusion' not in flags or '--use_adversarial' not in flags:
            continue
        need = [f for f in ('--adv_mode', '--lambda_r1') if f not in flags]
        if need:
            missing.append((name, need))
    assert not missing, (
        f'adversarial diffusion scenarios without critic stabilisers: {missing}. '
        f'Set --adv_mode hinge and --lambda_r1 > 0, or the critic saturates and '
        f'the run produces nothing usable.')


if __name__ == '__main__':
    rows = _scenarios()
    print(f'{len(rows)} scenarios in {SCENARIOS_SH.name}\n')
    failed = 0
    for name, flags in rows:
        try:
            _parse_flags(flags)
            print(f'  PASS  {name}')
        except SystemExit as e:
            failed += 1
            print(f'  FAIL  {name}  (argparse exit {e.code})')
            print(f'        {flags}')
    for check in (test_scenario_names_are_unique, test_no_retired_lambda_adv,
                  test_adversarial_diffusion_rows_carry_stabilisers):
        try:
            check()
            print(f'  PASS  {check.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {check.__name__}: {e}')
    print(f'\n{"ALL PASS" if not failed else f"{failed} FAILURE(S)"}')
    sys.exit(1 if failed else 0)
