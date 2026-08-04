"""
benchmark.py's run discovery, tiling lookup and paired-test guard.

WHY THIS EXISTS. A multi-phase run writes one manifest per phase SUBDIRECTORY
(`<run>/phase_infer/venous/manifest.csv`), because a multi-phase model emits
several volumes per case and they would collide on `{case_id}_syn.nii.gz` in one
directory. `discover()` looked only for the flat `<run>/phase_infer/manifest.csv`,
so `multiphase_film` and `multiphase_uncond` were absent from every benchmark
table — not reported as skipped, just gone.

Three failures, all silent, all of the kind this project cares most about:

  1. discover()    multi-phase runs missing from the table entirely.
  2. read_tiling() `manifest.parent.parent` is `<run>` for a flat manifest but
                   `<run>/phase_infer` for a per-phase one, so the config was not
                   found and `seam` came out NaN — reading as "external model, no
                   tiling geometry" rather than as a bug.
  3. paired_block() `paired_t([])` returns (0.0, 0.0, 0), so an arterial arm
                   compared against a venous baseline printed a full block of
                   '+0.000 ns' rows that read as "identical to the baseline" when
                   in fact no cases were comparable at all.
  4. select_baseline() with nothing scored raised a bare StopIteration. The
                   manifests are found, then every case fails to load (their paths
                   are stored relative to the repo root), every model is skipped,
                   and the user gets a traceback naming nothing. This is the normal
                   outcome of running benchmark.py from the wrong directory.

Run directly:
    python tests/test_benchmark_discovery.py
"""

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import shutil
import tempfile

import benchmark as B

FAILS = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)


def _make_tree(root: pathlib.Path):
    """A runs_dir with one of each layout the benchmark has to handle."""
    cfg = json.dumps({'patch_size': 128, 'overlap': 0.5})

    # flat, single-phase
    d = root / 'literature_baseline_l1_only'
    (d / 'phase_infer').mkdir(parents=True)
    (d / 'run_config.json').write_text(cfg)
    (d / 'phase_infer' / 'manifest.csv').write_text('gen_path\n')

    # multi-phase: one manifest per phase subdirectory
    d = root / 'literature_baseline_multiphase_film'
    for ph in ('venous', 'arterial'):
        (d / 'phase_infer' / ph).mkdir(parents=True)
        (d / 'phase_infer' / ph / 'manifest.csv').write_text('gen_path\n')
    (d / 'run_config.json').write_text(cfg)

    # trained but never inferred → must be REPORTED, not silently dropped
    d = root / 'literature_baseline_notinferred'
    d.mkdir(parents=True)
    (d / 'best_model.pth').write_text('x')
    (d / 'run_config.json').write_text(cfg)

    # external model: a manifest but no run_config anywhere → seam must be NaN
    d = root / 'external_baseline'
    (d / 'phase_infer').mkdir(parents=True)
    (d / 'phase_infer' / 'manifest.csv').write_text('gen_path\n')

    # a stray file in runs_dir must not crash the scan
    (root / 'notes.txt').write_text('x')


def test_discover():
    print('\ndiscover()')
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        _make_tree(root)
        found = B.discover(root)

        check('flat single-phase run found', 'l1_only' in found)
        check('multi-phase run found as one model PER PHASE',
              {'multiphase_film/venous', 'multiphase_film/arterial'} <= set(found),
              f'{sorted(k for k in found if "multiphase" in k)}')
        # Pooling the phases would average an arterial model's featHU into the
        # venous comparison and make the multi-phase arms incomparable to every
        # single-phase run — which is the entire point of the M1/M2/M3 design.
        check('phases are NOT pooled into one row', 'multiphase_film' not in found)
        check('external model (no run_config) still found', 'external_baseline' in found)
        check('un-inferred run is not scored', 'notinferred' not in found)
        check('stray file does not break the scan', len(found) == 4, f'{len(found)}')

        for name, m in found.items():
            check(f'{name}: manifest path exists', m.exists())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_tiling():
    print('\nread_tiling()')
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        _make_tree(root)
        found = B.discover(root)

        flat = B.read_tiling(found['l1_only'])
        check('flat layout resolves the config',
              flat == {'patch_size': 128, 'overlap': 0.5}, f'{flat}')

        for ph in ('venous', 'arterial'):
            t = B.read_tiling(found[f'multiphase_film/{ph}'])
            check(f'per-phase layout resolves the config ({ph})',
                  t == {'patch_size': 128, 'overlap': 0.5}, f'{t}')

        # An external model genuinely has no tiling geometry, and NaN seam is the
        # right answer there. The bug was that multi-phase runs looked like this.
        check('no run_config anywhere → {} (seam NaN is correct)',
              B.read_tiling(found['external_baseline']) == {})

        # Bounded walk: a config far above must not be picked up, or seam would be
        # scored against a tiling geometry from an unrelated run.
        (root / 'run_config.json').write_text(json.dumps({'patch_size': 999,
                                                          'overlap': 0.9}))
        check('does not reach a config outside the run dir',
              B.read_tiling(found['external_baseline']) == {})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _rows(keys):
    return [{'_key': k, 'psnr': 30.0, 'ssim': 0.94, 'org_ssim': 0.96, 'org_mae': 0.03,
             'feature_l1_hu': 14.0, 'raps_hf': 0.84, 'grad_w1': 0.002,
             'org_grad_w1': 0.006, 'seam': 1.35, 'zflicker': 0.9} for k in keys]


def test_paired_block_disjoint_cases():
    print('\npaired_block() with no shared cases')
    out = []
    B.paired_block({'base': _rows(['/venous/a', '/venous/b', '/venous/c']),
                    'mp/venous': _rows(['/venous/a', '/venous/b', '/venous/c']),
                    'mp/arterial': _rows(['/arterial/a', '/arterial/b'])},
                   'base', out)
    txt = '\n'.join(out)

    check('arterial arm is flagged not comparable',
          'no cases in common' in txt)
    # The failure mode: a full block of zero-delta rows that reads as "identical
    # to the baseline" when nothing was compared.
    arterial_lines = [l for l in out if l.startswith('mp/arterial')]
    check('arterial arm emits no fake zero-delta rows',
          not any('+0.000' in l for l in arterial_lines),
          f'{len(arterial_lines)} line(s)')
    check('venous arm still gets a real comparison block',
          any(l.startswith('mp/venous') and 'feature_l1_hu' in l for l in out))
    check('model column is wide enough for a <run>/<phase> name',
          all('mp/arterialfeature' not in l and 'mp/venousfeature' not in l
              for l in out))


def test_select_baseline():
    print('\nselect_baseline()')

    # The failure this guards: nothing scored. Must be a SystemExit carrying a
    # message, never the bare StopIteration that an eagerly-evaluated next()
    # default produces.
    try:
        B.select_baseline({}, None)
        check('empty table raises', False, 'returned instead of raising')
    except SystemExit as e:
        check('empty table raises SystemExit with a message', bool(str(e)), f'{e}')
    except StopIteration:
        check('empty table raises SystemExit, not StopIteration', False)

    rows = {'l1_only': _rows(['a']), 'diff_v': _rows(['a'])}
    check('prefers an *_only run as the default baseline',
          B.select_baseline(rows, None) == 'l1_only')
    check('explicit baseline is honoured',
          B.select_baseline(rows, 'diff_v') == 'diff_v')
    # discover() strips the prefix, so the name a user reads off `ls` differs from
    # the table key. Both spellings must resolve or --baseline silently misses.
    check('run-directory spelling resolves',
          B.select_baseline(rows, 'literature_baseline_diff_v') == 'diff_v')

    try:
        B.select_baseline(rows, 'nosuchmodel')
        check('unknown baseline raises', False, 'returned instead of raising')
    except SystemExit as e:
        check('unknown baseline names the available models',
              'l1_only' in str(e) and 'diff_v' in str(e), f'{e}')

    # No *_only run and no request: fall back to some model rather than raising.
    check('falls back to the first model when no *_only run exists',
          B.select_baseline({'diff_v': _rows(['a']), 'diff_x0': _rows(['a'])}, None)
          in ('diff_v', 'diff_x0'))


if __name__ == '__main__':
    print('=' * 70)
    print('BENCHMARK DISCOVERY / TILING / PAIRED TESTS')
    print('=' * 70)
    test_discover()
    test_read_tiling()
    test_paired_block_disjoint_cases()
    test_select_baseline()
    print('\n' + '=' * 70)
    if FAILS:
        print(f'FAILED ({len(FAILS)}): {", ".join(FAILS)}')
        sys.exit(1)
    print('ALL PASS')
