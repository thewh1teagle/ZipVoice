# Recursive Flow Matching for ZipVoice

Based on the RecFM paper: https://arxiv.org/pdf/2605.26535v1

This note describes the core RecFM adaptation for ZipVoice. It only covers the
model and training idea, not dataset preparation, shell scripts, checkpoint
averaging, or deployment.

## Existing ZipVoice Flow

ZipVoice trains a conditional flow-matching decoder on acoustic features. In the
current code, time is oriented as:

```python
xt = features * t + noise * (1 - t)
target_v = features - noise
```

So `t = 0` is noise and `t = 1` is the target fbank feature. Inference samples
noise and integrates the learned velocity field with Euler steps.

ZipVoice handles CFG in the sampler wrapper by making conditional and
unconditional decoder calls. RecFM does not need to change that mechanism; it
only adds a recursive-scale conditioning signal to the velocity network.

## RecFM Mapping

Depth-2 RecFM trains the same noisy point `xt` under two aligned trajectory
parameterizations:

```python
t ~ Uniform(0, 1)
alpha ~ Uniform(t, 1)
tau = t / alpha

primary_v = model(xt, t, alpha=1)
secondary_v = model(xt, tau, alpha=alpha)
```

The target for the primary trajectory is the normal ZipVoice velocity:

```python
target_v = features - noise
```

The secondary trajectory is trained to represent a scaled version of the same
direction:

```python
secondary_target_v = alpha * target_v
```

The total RecFM objective is:

```python
primary_loss = mse(primary_v, target_v)
secondary_loss = mse(secondary_v, alpha * target_v)
consistency_loss = mse(secondary_v, alpha * primary_v)

loss = primary_loss + secondary_loss + lambda * consistency_loss
```

All losses are applied only on the same speech-generation mask used by the
existing ZipVoice fbank loss.

## Core Code Changes

The decoder needs one new scalar conditioning path:

- keep the existing time embedding `t`
- add a new `recfm_scale` embedding for the RecFM trajectory scale `alpha`

At inference, a RecFM model should use the primary trajectory:

```python
recfm_scale = 1.0
```

Then normal Euler sampling can be reused. The intended gain is that training has
explicitly regularized large-step velocity consistency, making 2-step inference
more plausible than simply lowering `num_step` on an ordinary ZipVoice-Distill
checkpoint.

## Important Constraints

This is not a sampler-only change. A checkpoint must be fine-tuned or trained
with the RecFM objective so the new `recfm_scale` embedding and consistency
behavior are learned.

A clean implementation should start from `zipvoice` weights, load them with
`strict=False`, and randomly initialize only the new RecFM scale embedding.
Training only the flow decoder is the conservative default because the text
encoder and duration/text conditioning do not need to change.

The realistic target is 2-step inference. One-step inference may work after
tuning, but it asks the model to solve the full noise-to-feature path in a
single evaluation and is much less likely to preserve TTS quality.
