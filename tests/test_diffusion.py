"""
Properties of the diffusion stack that must hold, checked without data or a GPU.

Every test here pins something a silent bug would break invisibly — a wrong
schedule still trains, a decorrelated noise field still produces plausible
images, a saturating output head still converges to something. These are the
checks that would catch it.

Run directly (the project convention — there is no CI):
    python tests/test_diffusion.py
"""

# Repo modules live one level up (this file sits in scripts/ or tests/).
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from models_diffusion import (DDIMSampler, DiffusionUNet, NoiseSchedule,
                              PARAMETERISATIONS, from_model, to_model)

FAILS = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------

def test_schedule_roundtrip():
    """to_target then predict_x0 must recover x0 exactly, at every t.

    This is the algebra the whole sampler rests on. A sign error or a swapped
    sqrt_ab/sqrt_1mab would still produce a trainable model — it would just
    converge to the wrong thing — so it is checked in closed form rather than
    inferred from sample quality.
    """
    print('\nschedule round-trip')
    sch = NoiseSchedule(1000)
    torch.manual_seed(0)
    x0 = torch.rand(4, 1, 16, 16) * 2 - 1
    eps = torch.randn_like(x0)
    for param in PARAMETERISATIONS:
        errs_x0, errs_eps = [], []
        for tv in (0, 1, 250, 500, 750, 999):
            t = torch.full((4,), tv, dtype=torch.long)
            x_t = sch.q_sample(x0, t, eps)
            tgt = sch.to_target(x0, eps, t, param)
            errs_x0.append(float((sch.predict_x0(tgt, x_t, t, param, clip=False) - x0).abs().max()))
            errs_eps.append(float((sch.predict_eps(tgt, x_t, t, param) - eps).abs().max()))
        check(f'{param}: predict_x0 recovers x0', max(errs_x0) < 1e-4,
              f'max err {max(errs_x0):.2e}')
        # eps at t=0 is recovered by dividing by sqrt(1-alpha_bar), which is
        # ~0 there, so the tolerance has to allow for that amplification.
        check(f'{param}: predict_eps recovers eps', max(errs_eps) < 1e-2,
              f'max err {max(errs_eps):.2e}')


def test_schedule_monotone():
    """alpha_bar must fall monotonically from ~1 to ~0 over the schedule."""
    print('\nschedule shape')
    sch = NoiseSchedule(1000)
    ab = sch.alphas_cumprod
    check('alpha_bar decreasing', bool((ab[1:] <= ab[:-1] + 1e-7).all()))
    check('alpha_bar starts near 1', float(ab[0]) > 0.99, f'{float(ab[0]):.4f}')
    check('alpha_bar ends near 0', float(ab[-1]) < 0.01, f'{float(ab[-1]):.2e}')
    # Cosine vs linear: at the halfway point a cosine schedule still leaves real
    # signal, which is the whole reason it was chosen (§ models_diffusion).
    check('alpha_bar mid retains signal', 0.05 < float(ab[500]) < 0.6,
          f'{float(ab[500]):.3f}')


def test_domain_conversion():
    """[0,1] <-> [-1,1] must round-trip and must actually re-centre."""
    print('\ndomain conversion')
    x = torch.rand(8, 1, 8, 8)
    check('to_model/from_model round-trip',
          torch.allclose(from_model(to_model(x)), x, atol=1e-6))
    check('to_model centres on 0', abs(float(to_model(x).mean())) < 0.2)


def test_output_head_unbounded():
    """The output head must NOT be a Sigmoid.

    v and eps are unbounded and symmetric about zero; a Sigmoid head would clip
    half the target range. Checked by looking at the actual output range with
    large-magnitude weights rather than by inspecting the module list, so it
    stays true if the head is ever restructured.
    """
    print('\noutput head')
    torch.manual_seed(0)
    m = DiffusionUNet(dims=2, base_channels=8, cond_channels=1)
    with torch.no_grad():
        m.out_conv.weight.mul_(50.0)
        m.out_conv.bias.fill_(-3.0)
    m.eval()
    out = m(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32),
            torch.zeros(2, dtype=torch.long))
    check('output can go negative', float(out.min()) < 0.0, f'min {float(out.min()):.2f}')
    check('output not capped at 1', float(out.max()) > 1.0 or float(out.min()) < -1.0,
          f'range [{float(out.min()):.2f}, {float(out.max()):.2f}]')


def test_film_zero_init():
    """At init every FiLM site is an identity, so the output is t-independent.

    Same guarantee `tests/test_phase_cond.py` pins for phase conditioning. It
    means a diffusion run starts as a plain U-Net on cat([x_t, ncct]) and the
    conditioning has to be learned rather than assumed — and it makes a nonzero
    gamma later in `history.json` a result rather than an artifact of init.
    """
    print('\nFiLM zero-init')
    torch.manual_seed(0)
    m = DiffusionUNet(dims=2, base_channels=8, cond_channels=1); m.eval()
    x, c = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
    o_lo = m(x, c, torch.zeros(2, dtype=torch.long))
    o_hi = m(x, c, torch.full((2,), 999, dtype=torch.long))
    check('output independent of t at step 0', torch.allclose(o_lo, o_hi, atol=1e-6))
    g = m.film_stats()
    check('all gammas start at 0', max(abs(v) for v in g.values()) < 1e-8)
    check('film_stats covers encoder too', 'gamma_enc1' in g and 'gamma_dec1' in g,
          f'{len(g)} sites')


def test_cfg_null_path():
    """Dropping the condition must make the output independent of the NCCT, and
    eval must never drop silently."""
    print('\nclassifier-free guidance')
    torch.manual_seed(0)
    m = DiffusionUNet(dims=2, base_channels=8, cond_channels=1); m.eval()
    x = torch.randn(2, 1, 32, 32)
    t = torch.zeros(2, dtype=torch.long)
    allm = torch.ones(2, dtype=torch.bool)
    o1 = m(x, torch.randn(2, 1, 32, 32), t, drop_cond=allm)
    o2 = m(x, torch.randn(2, 1, 32, 32), t, drop_cond=allm)
    check('null path ignores the NCCT', torch.allclose(o1, o2, atol=1e-6))
    # In eval with drop_cond=None nothing may be dropped, or every reported
    # metric would be a partly-unconditional model's.
    c = torch.randn(2, 1, 32, 32)
    check('eval never drops silently', not torch.allclose(m(x, c, t), o1, atol=1e-6))


def test_ddim_determinism():
    """eta=0 must be a deterministic function of x_T; eta=1 must not be.

    The volume-wide noise field in infer_volume.py is only coherent because of
    the first half of this. If DDIM at eta=0 ever became stochastic, tiles would
    decorrelate and the variance-ratio result would be meaningless.
    """
    print('\nDDIM determinism')
    torch.manual_seed(0)
    m = DiffusionUNet(dims=2, base_channels=8, cond_channels=1); m.eval()
    sch = NoiseSchedule(1000)
    cond, x_T = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)

    s0 = DDIMSampler(m, sch, n_steps=8, eta=0.0)
    a, b = s0.sample(cond, x_T), s0.sample(cond, x_T)
    check('eta=0 is deterministic', torch.equal(a, b))
    check('eta=0 depends on x_T',
          not torch.allclose(a, s0.sample(cond, torch.randn_like(x_T)), atol=1e-4))
    check('sample stays in [-1,1]', float(a.min()) >= -1.001 and float(a.max()) <= 1.001,
          f'[{float(a.min()):.3f}, {float(a.max()):.3f}]')

    s1 = DDIMSampler(m, sch, n_steps=8, eta=1.0)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    check('eta=1 reproducible from its generator',
          torch.equal(s1.sample(cond, x_T, generator=g1),
                      s1.sample(cond, x_T, generator=g2)))
    check('eta=1 differs from eta=0',
          not torch.equal(s1.sample(cond, x_T,
                                    generator=torch.Generator().manual_seed(3)), a))

    check('guidance changes the output',
          not torch.equal(DDIMSampler(m, sch, n_steps=8, eta=0.0,
                                      guidance=3.0).sample(cond, x_T), a))
    ts = s0.timesteps(torch.device('cpu'))
    check('timesteps descend and end at 0',
          bool((ts[1:] < ts[:-1]).all()) and int(ts[-1]) == 0, f'{ts.tolist()}')


def test_volume_noise_coherence():
    """Two overlapping tiles cropped from ONE field must agree on the overlap.

    This is the property DIFFUSION_PLAN.md §6 step 6 exists to guarantee, and
    the one an "obvious" per-tile torch.randn silently destroys. It is checked
    against DiffusionPredictor's real cropping code, not a reimplementation.
    """
    print('\nvolume-wide noise field')
    from infer_volume import DiffusionPredictor
    torch.manual_seed(0)
    m = DiffusionUNet(dims=2, base_channels=8, cond_channels=1); m.eval()
    cfg = {'patch_depth': 1, 'patch_size': 32}
    p = DiffusionPredictor(m, NoiseSchedule(1000), 'cpu', False, cfg,
                           sample_seed=5, case_id='case_A')
    p.begin_volume((4, 64, 64))

    # Two tiles at stride 16 → a 16-column overlap.
    a = p._x_T([(1, 0, 0)])[0, 0]
    b = p._x_T([(1, 0, 16)])[0, 0]
    check('overlapping tiles share noise', torch.equal(a[:, 16:], b[:, :16]))
    check('disjoint tiles differ', not torch.equal(a, p._x_T([(1, 32, 32)])[0, 0]))

    # Reproducible from (seed, case, k), and different across k and across cases.
    p2 = DiffusionPredictor(m, NoiseSchedule(1000), 'cpu', False, cfg,
                            sample_seed=5, case_id='case_A')
    p2.begin_volume((4, 64, 64))
    check('same (seed, case, k) reproduces the field',
          torch.equal(p._noise, p2._noise))
    p2.set_sample(1)
    check('sample index changes the field', not torch.equal(p._noise, p2._noise))
    p3 = DiffusionPredictor(m, NoiseSchedule(1000), 'cpu', False, cfg,
                            sample_seed=5, case_id='case_B')
    p3.begin_volume((4, 64, 64))
    check('case id changes the field', not torch.equal(p._noise, p3._noise))

    # An independent-per-tile field would fail the first check; assert that the
    # naive alternative really is different, so the test cannot pass vacuously.
    check('naive per-tile noise would NOT be coherent',
          not torch.equal(torch.randn(32, 32), torch.randn(32, 32)))


def test_dims3():
    """The whole stack must run at dims=3.

    The 3-D paths were kept in this fork specifically because the rank-agnostic
    broadcasts are what let FiLM(t) drop in unchanged. If they rot, that claim
    stops being true and nobody notices until a 3-D run is attempted.
    """
    print('\ndims=3')
    torch.manual_seed(0)
    m = DiffusionUNet(dims=3, base_channels=8, cond_channels=1,
                      pool_stride=(1, 2, 2)); m.eval()
    sch = NoiseSchedule(200)
    x0 = torch.rand(1, 1, 4, 32, 32) * 2 - 1
    eps = torch.randn_like(x0)
    t = torch.full((1,), 100, dtype=torch.long)
    x_t = sch.q_sample(x0, t, eps)
    check('3-D forward', m(x_t, torch.randn_like(x0), t).shape == x0.shape)
    check('3-D schedule broadcast', x_t.shape == x0.shape)
    out = DDIMSampler(m, sch, n_steps=4, eta=0.0).sample(torch.randn_like(x0), eps)
    check('3-D sampling', out.shape == x0.shape and bool(torch.isfinite(out).all()))


def test_train_step_shapes():
    """One optimiser step end-to-end on random tensors, both parameterisations."""
    print('\ntraining step')
    import torch.nn.functional as F
    for param in PARAMETERISATIONS:
        torch.manual_seed(0)
        m = DiffusionUNet(dims=2, base_channels=8, cond_channels=1,
                          parameterisation=param)
        sch = NoiseSchedule(1000)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        src, tgt = torch.rand(4, 1, 32, 32), torch.rand(4, 1, 32, 32)
        before = float(m.film_dec1.to_scale_shift[-1].weight.abs().sum())
        for _ in range(2):
            t = torch.randint(0, 1000, (4,))
            noise = torch.randn_like(tgt)
            x0 = to_model(tgt)
            x_t = sch.q_sample(x0, t, noise)
            loss = F.mse_loss(m(x_t, to_model(src), t),
                              sch.to_target(x0, noise, t, param))
            opt.zero_grad(); loss.backward(); opt.step()
        after = float(m.film_dec1.to_scale_shift[-1].weight.abs().sum())
        check(f'{param}: loss is finite', np.isfinite(float(loss)))
        # The FiLM head starts at exactly zero; if it is still zero after two
        # steps, no gradient reaches the timestep conditioning at all.
        check(f'{param}: gradient reaches FiLM(t)', after > before,
              f'{before:.2e} -> {after:.2e}')


if __name__ == '__main__':
    print('=' * 70)
    print('DIFFUSION STACK')
    print('=' * 70)
    test_schedule_roundtrip()
    test_schedule_monotone()
    test_domain_conversion()
    test_output_head_unbounded()
    test_film_zero_init()
    test_cfg_null_path()
    test_ddim_determinism()
    test_volume_noise_coherence()
    test_dims3()
    test_train_step_shapes()
    print('\n' + '=' * 70)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print('ALL PASS')
