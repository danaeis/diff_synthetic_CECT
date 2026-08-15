"""
Properties of the adversarial branch that must hold, checked without data or a GPU.

Every one of these pins something that a GAN would keep training through. A
discriminator that never sees t, a gradient that never reaches the generator, a
clamp that silently zeroes the gradient on the voxels it was meant to fix, a
critic that separates real from fake on batch statistics — none of these raise,
none of them show up in the loss curves as anything but "the adversarial term
did not help". So they are checked in closed form instead.

Run directly (the project convention — there is no CI):
    python tests/test_adversarial.py
"""

import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.ERROR)

import torch

from config import train_config
from losses import AdversarialLoss, FeatureMatchingLoss
from models_disc import PatchGANDiscriminator
from trainer import Trainer
from trainer_diffusion import DiffusionTrainer

FAILS = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)


def _cfg(**over):
    c = dict(train_config)
    c.update(dict(device='cpu', output_dir=pathlib.Path(tempfile.mkdtemp()),
                  dims=2, patch_depth=1, patch_size=32,
                  generator_base_channels=8, use_mixed_precision=False,
                  diffusion_steps=100, report_per_organ_metrics=False,
                  use_adversarial=True, use_cond_disc=True, use_phase_cond=True,
                  adv_warmup_epochs=2))
    c.update(over)
    return c


def _batch(B=8, C=1, dims=2, S=32, D=8):
    shp = (B, 1, D, S, S) if dims == 3 else (B, 1, S, S)
    src = (B, C, D, S, S) if dims == 3 else (B, C, S, S)
    return {'source': torch.rand(*src), 'target': torch.rand(*shp),
            'mask': torch.randint(0, 3, shp).float(),
            'phase': torch.randint(0, 2, (B,))}


# ---------------------------------------------------------------------------

def test_d_is_t_conditioned():
    """D must give two different verdicts on the SAME image at two timesteps.

    This is the property that separates a diffusion critic from a texture
    critic. Without it the cheapest solution for D is "blurry => fake", i.e. a
    timestep regressor, and the only way the generator can beat that is by
    hallucinating detail at high t — the exact failure the term is supposed to
    prevent. It is invisible in training: an unconditional D produces a perfectly
    healthy-looking pair of loss curves.
    """
    print('\ntimestep conditioning')
    torch.manual_seed(0)
    D = PatchGANDiscriminator(dims=2, ndf=16, cond_dim=32, use_t_cond=True)
    # cond_proj is zero-initialised, so an untrained D is deliberately blind to t
    # (a warm start, not a dead branch). Give it one gradient step first.
    torch.nn.init.normal_(D.cond_proj.weight, std=0.5)
    img = torch.rand(4, 1, 32, 32) * 2 - 1
    with torch.no_grad():
        lo = D(img, t=torch.zeros(4, dtype=torch.long))[0]
        hi = D(img, t=torch.full((4,), 99, dtype=torch.long))[0]
    diff = float((lo - hi).abs().max())
    check('D(t=0) != D(t=99) on identical images', diff > 1e-3,
          f'max|diff| = {diff:.4f}')

    D0 = PatchGANDiscriminator(dims=2, ndf=16)
    try:
        D0(img, t=torch.zeros(4, dtype=torch.long))
        check('unconditional D rejects a `t`', False, 'no error raised')
    except ValueError:
        check('unconditional D rejects a `t`', True)


def test_no_batchnorm_leak():
    """The default D must give the same verdict on an image whether it is scored
    alone or alongside others.

    The reference D uses BatchNorm and is called on the real batch and the fake
    batch in SEPARATE forwards, so each is normalised by its own statistics: D
    can separate real from fake without looking at the content at all. GroupNorm
    is per-sample and cannot. Checked as batch-independence, which is what the
    property actually is.
    """
    print('\nreal/fake batch-statistics leak')
    torch.manual_seed(0)
    x = torch.rand(4, 1, 32, 32) * 2 - 1
    for norm, want in [('group', True), ('batch', False)]:
        D = PatchGANDiscriminator(dims=2, ndf=16, norm=norm, spectral=False).eval()
        # train() mode is where BatchNorm uses the batch's own statistics.
        D.train()
        with torch.no_grad():
            alone = D(x[:1])[0]
            # Same first item, very different companions.
            together = D(torch.cat([x[:1], x[1:] * 0.05 + 0.9]))[0][:1]
        diff = float((alone - together).abs().max())
        check(f"norm={norm}: verdict independent of batch-mates "
              f"({'expected' if want else 'expected NOT'})",
              (diff < 1e-4) is want, f'max|diff| = {diff:.5f}')


def test_straight_through_clip():
    """The bound D sees must clamp the VALUE without killing the gradient.

    Real x0 is in [-1,1] and an unclipped x0_hat is not, so an unbounded fake
    hands D a discriminating feature that is about range rather than realism. A
    plain clamp fixes that and zeroes the gradient on exactly the saturated
    voxels that are out of range — the mistake this repo already made once with
    the organ losses (see AUX_MAX_T in config.py).
    """
    print('\nstraight-through bound on x0_hat')
    T = DiffusionTrainer(_cfg(use_diffusion=True))
    x = torch.tensor([-3.0, -0.5, 0.5, 4.0], requires_grad=True)
    y = T._adv_bound(x)
    y.sum().backward()
    check('value is clamped to [-1,1]',
          float(y.max()) <= 1.0 and float(y.min()) >= -1.0, str(y.tolist()))
    check('gradient survives on saturated voxels',
          x.grad.tolist() == [1.0] * 4, str(x.grad.tolist()))

    T.adv_clip_mode = 'hard'
    x2 = torch.tensor([-3.0, -0.5, 0.5, 4.0], requires_grad=True)
    T._adv_bound(x2).sum().backward()
    check("'hard' reproduces the trap it replaces (for ablation)",
          x2.grad.tolist() == [0.0, 1.0, 1.0, 0.0], str(x2.grad.tolist()))


def test_gradient_reaches_generator():
    """The adversarial term must change the generator's gradient, and must NOT
    leave gradient on the discriminator's parameters.

    Both halves fail silently. A detached fake still trains D perfectly happily
    while giving G nothing; an unfrozen D accumulates generator-scaled gradients
    between its own updates, which is one reordering away from training the
    critic on the wrong objective.
    """
    print('\ngradient routing')
    torch.manual_seed(0)
    T = DiffusionTrainer(_cfg(use_diffusion=True, adv_max_t=100))
    T.current_epoch = 5
    b = _batch()
    src, tgt, ph = b['source'], b['target'], T._phase(b)
    t = torch.randint(0, 50, (8,))
    noise = torch.randn_like(tgt)

    def gnorm(with_adv):
        T.opt_G.zero_grad(set_to_none=True)
        loss, _, x0 = T._diffusion_loss(src, tgt, None, ph, None, t, noise)
        if with_adv:
            T.disc_freq, T.global_step = 2, 1        # isolate the G backward
            term, _ = T._adversarial_term(src, tgt, x0, t, ph, None)
            T.disc_freq = 1
            loss = loss + term
        loss.backward()
        return sum(p.grad.norm().item() ** 2
                   for p in T.G.parameters() if p.grad is not None) ** 0.5

    for p in T.D.parameters():
        p.grad = None
    plain = gnorm(False)
    T.opt_G.zero_grad(set_to_none=True)
    for p in T.D.parameters():
        p.grad = None
    with_adv = gnorm(True)
    check('adversarial term changes |grad G|', abs(with_adv - plain) > 1e-6,
          f'{plain:.4f} -> {with_adv:.4f}')
    check('D holds no gradient after the G backward',
          not [p for p in T.D.parameters() if p.grad is not None])
    check('D is unfrozen again after the step',
          all(p.requires_grad for p in T.D.parameters()))


def test_t_gate_and_shared_conditioning():
    """Only items with t < adv_max_t reach D, and real/fake get the SAME t.

    Handing D one t for the real branch and another for the fake branch would let
    it separate them on the conditioning rather than the image — the failure that
    conditioning exists to avoid.
    """
    print('\ntimestep gate')
    torch.manual_seed(0)
    T = DiffusionTrainer(_cfg(use_diffusion=True, adv_max_t=50))
    T.current_epoch = 5
    b = _batch()
    src, tgt, ph = b['source'], b['target'], T._phase(b)
    t = torch.arange(0, 100, 100 // 8)[:8]
    noise = torch.randn_like(tgt)
    _, _, x0 = T._diffusion_loss(src, tgt, None, ph, None, t, noise)

    seen = []
    orig = T._disc_step
    T._disc_step = lambda real, fake, source=None, t=None, cond=None: (
        seen.append(t.clone()) or orig(real, fake, source=source, t=t, cond=cond))
    T.global_step = 0
    T._adversarial_term(src, tgt, x0.detach(), t, ph, None)
    check('D saw exactly the t < adv_max_t subset',
          bool(seen) and torch.equal(seen[0], t[t < 50]),
          f'{seen[0].tolist() if seen else None} from {t.tolist()}')

    # Nothing below the gate → no term, and the log channels report NaN, which
    # is this codebase's "not measured" sentinel. Reporting 0.0 would average in
    # as a measured zero and scale train_adv by the gate's firing rate.
    import math
    T.global_step = 0
    term, log = T._adversarial_term(src, tgt, x0.detach(),
                                    torch.full((8,), 99), ph, None)
    check('empty gate skips the term and reports NaN, not 0.0',
          term is None and log['n_adv'] == 0
          and all(math.isnan(log[k]) for k in ('adv', 'fm', 'disc')))

    # And the epoch mean must then be over the steps that DID fire.
    vals = [1.0, float('nan'), 3.0]
    import numpy as np
    check('nanmean ignores un-measured steps', float(np.nanmean(vals)) == 2.0)


def test_lambda_adv_has_one_owner():
    """The warmed-up lambda the loss APPLIES must equal the one history RECORDS.

    Two schedules is the bug this is guarding: the reference kept the warmup
    inside the loss object, which only the deterministic path advances, while the
    trainer logged its own. They agree right up until one path stops calling
    set_epoch, and then history['lambda_adv'] describes a curriculum that never
    ran.
    """
    print('\nlambda_adv ownership')
    T = Trainer(_cfg(adv_warmup_epochs=4, lambda_adv=2.0))
    got = []
    for ep in (0, 1, 2, 4, 8):
        T.current_epoch = ep
        got.append(round(T._adv_w(), 3))
    check('warmup ramps linearly then holds', got == [0.0, 0.5, 1.0, 2.0, 2.0], str(got))
    check('CompositeLoss owns no second schedule',
          not hasattr(T.criterion, 'adv_warmup') and not hasattr(T.criterion, 'lambda_adv'))

    T.current_epoch = 1
    step = T._train_step(_batch())
    # train_adv is logged RAW; its contribution is train_adv * lambda_adv.
    check('the step reports both the raw loss and its weight',
          step['adv'] > 0 and abs(T._adv_w() - 0.5) < 1e-9,
          f"adv={step['adv']:.4f} lambda={T._adv_w():.2f}")


def test_flag_off_is_untouched():
    """With the flag off there must be no D, no adversarial keys in the loss, and
    no extra RNG draw — a non-adversarial run has to stay bit-identical to what
    it was before this branch existed."""
    print('\nflag off')
    torch.manual_seed(1234)
    a = Trainer(_cfg(use_adversarial=False, use_feature_matching=False))
    torch.manual_seed(1234)
    b = Trainer(_cfg(use_adversarial=True))
    check('no D is built', a.D is None)
    same = all(torch.equal(p, q) for p, q in
               zip(a.G.state_dict().values(), b.G.state_dict().values()))
    check('generator init is identical with and without D', same)
    step = a._train_step(_batch())
    check('step reports zeroed adversarial channels',
          step['adv'] == 0 and step['fm'] == 0 and step['disc'] == 0)


def test_feature_matching_edge_cases():
    print('\nfeature matching')
    fm = FeatureMatchingLoss()
    f = [torch.rand(2, 4, 8, 8), torch.rand(2, 8, 4, 4)]
    check('zero on identical features', float(fm(f, f)) == 0.0)
    # A bare torch.tensor(0.0) here is a latent crash the moment it is added to
    # a CUDA total; the empty case has to match the features it was given.
    z = fm([], f)
    check('empty real features → device/dtype-correct zero',
          float(z) == 0.0 and z.device == f[0].device and z.dtype == f[0].dtype)


def test_25d_diffusion_adversarial():
    """2.5-D input, on the diffusion path, with the critic on.

    Three channel counts have to agree and nothing raises if they do not — the
    U-Net would just be conditioned on a truncated stack, or D on the wrong
    pairing, and both still train. `tests/test_level_cond.py::test_25d` covers
    the deterministic path; this is the combination that has three separate
    consumers of `in_channels`.
    """
    print('\n2.5-D + diffusion + adversarial')
    n_in = 5
    T = DiffusionTrainer(_cfg(use_diffusion=True, n_input_slices=n_in,
                              in_channels=n_in, patch_depth=1, dims=2))
    T.current_epoch = 5
    g_in = next(T.G.enc1.parameters()).shape[1]
    check('U-Net takes 1 noisy + n_input_slices cond channels',
          g_in == 1 + n_in, f'{g_in} == 1 + {n_in}')
    check('the CFG null embedding matches the cond width',
          T.G.null_cond.shape[1] == n_in, str(tuple(T.G.null_cond.shape)))
    check('conditional D takes 1 + n_input_slices channels',
          T.D.in_channels == 1 + n_in, f'{T.D.in_channels} == 1 + {n_in}')

    b = _batch(B=8, C=n_in)
    step = T._train_step(b)
    check('a 2.5-D adversarial step runs and is finite',
          step['gen_total'] == step['gen_total'] and step['adv'] == step['adv'],
          f"total={step['gen_total']:.3f} adv={step['adv']:.3f}")

    # Inference has its own 2.5-D path (edge-clamped slice stacking + tiling);
    # a shape/finiteness round-trip is what catches a mis-stacked channel axis.
    from infer_volume import DiffusionPredictor, infer_volume
    import numpy as np
    cfg = _cfg(use_diffusion=True, n_input_slices=n_in, in_channels=n_in)
    vol = np.random.uniform(-50, 150, (7, 32, 32)).astype(np.float32)
    pred = DiffusionPredictor(T.G.eval(), T.schedule, 'cpu', False, cfg,
                              ddim_steps=3, phase_id=0, case_id='c0')
    res = infer_volume(pred, vol, cfg, 'cpu', batch_size=4)[0]
    check('diffusion inference round-trips a 2.5-D volume',
          res.shape == vol.shape and bool(np.isfinite(res).all()),
          f'{vol.shape} -> {res.shape}')
    T.G.train()


def test_modes_and_resume():
    print('\nGAN modes and checkpoint round-trip')
    for mode in ('lsgan', 'bce', 'hinge'):
        a = AdversarialLoss(mode)
        r, f = torch.randn(4, 1, 8, 8), torch.randn(4, 1, 8, 8)
        dl, gl = a.disc_loss(r, f), a.gen_loss(f)
        check(f'{mode}: finite D and G losses',
              torch.isfinite(dl) and torch.isfinite(gl),
              f'D={float(dl):.3f} G={float(gl):.3f}')
    # A perfect critic must score better than a confused one, in every mode.
    for mode in ('lsgan', 'bce', 'hinge'):
        a = AdversarialLoss(mode)
        good = a.disc_loss(torch.full((4, 1, 4, 4), 3.0), torch.full((4, 1, 4, 4), -3.0))
        bad = a.disc_loss(torch.full((4, 1, 4, 4), -3.0), torch.full((4, 1, 4, 4), 3.0))
        check(f'{mode}: a correct D scores lower than an inverted one',
              float(good) < float(bad), f'{float(good):.3f} < {float(bad):.3f}')

    T = DiffusionTrainer(_cfg(use_diffusion=True, use_feature_matching=True,
                              lambda_r1=1.0, r1_every=1))
    T.current_epoch = 5
    T._train_step(_batch())
    T._save_checkpoint(1, False)
    ck = sorted(pathlib.Path(T.out).glob('ckpt_ep*.pth'))[0]
    st = torch.load(ck, map_location='cpu', weights_only=False)
    check('checkpoint carries D, its optimiser and its step count',
          'D_state' in st and 'opt_D' in st and 'd_steps' in st)
    T2 = DiffusionTrainer(_cfg(use_diffusion=True, use_feature_matching=True,
                               lambda_r1=1.0, r1_every=1))
    T2.load_checkpoint(str(ck))
    check('D round-trips exactly',
          all(torch.equal(p, q) for p, q in
              zip(T.D.state_dict().values(), T2.D.state_dict().values())))


if __name__ == '__main__':
    test_d_is_t_conditioned()
    test_no_batchnorm_leak()
    test_straight_through_clip()
    test_gradient_reaches_generator()
    test_t_gate_and_shared_conditioning()
    test_lambda_adv_has_one_owner()
    test_flag_off_is_untouched()
    test_feature_matching_edge_cases()
    test_25d_diffusion_adversarial()
    test_modes_and_resume()
    print('\n' + '=' * 70)
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print('ALL PASS')
