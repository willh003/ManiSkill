# Peg hidden hole: language feedback vs the final-error vector (strong expert)

Same setup as `2026-10-27-peg-hidden-strong-expert.md` (hidden hole, student's own episodes, strong expert ckpt_3576 with 1.00 success, 30 iters,
512 tasks, k=4, 2000 grad steps/iter); only the feedback differs. After each episode the feedback is a templated command computed from the final
error: "shift the peg 4.2 cm forward and 1.5 cm right and 0.3 cm down then turn it 12 degrees clockwise" (3 openers x 6 axis orders x 2 rotation
verbs; whole mm and degrees; forward/left/up = +x/+y/+z world; rotation = hole yaw - peg yaw). Code: `command_tokens` in `icl_tasks.py`, feedback kinds
`language` / `language_pretrained` in `icl_episodic.py` (see `implementations/icl-code-layout.md`).
- `language`: word ids (31-word vocab, digits one token each) -> from-scratch `nn.Embedding` + word position, up to 32 tokens per episode in the context.
- `language_pretrained`: the command string -> frozen `all-MiniLM-L6-v2` (384-d) -> linear -> one feedback token.
- `language_tokens`: the command text -> frozen `google/flan-t5-small` encoder, one 512-d vector per word piece (<=36, fp16 + mask) -> LayerNorm + linear -> context tokens; no hand-built vocabulary.
Runs `logs/runs/icl-peg-{language,language-pretrained,language-tokens}-hidden-strong` (jobs 42466737 on l40, 42467948 and 42479042 on l40s). Plot: `logs/renders/icl-peg/peg_language.png`.

| feedback (success_once) | h0 | h1 | h2 | h3 | h4 | shuf_y h1 | no_y h1 | no_ctx h1 |
|---|---|---|---|---|---|---|---|---|
| final error vector (true hole pose) | .07 | 1.00 | .99 | .99 | 1.00 | .02 | .00 | .07 |
| language, from scratch | .07 | .84 | .90 | .89 | .88 | .01 | .03 | .07 |
| language, frozen MiniLM (pooled) | .06 | .27 | .42 | .48 | .45 | .02 | .02 | .07 |
| language, frozen flan-T5-small (per word piece) | .07 | .81 | .83 | .87 | .82 | .00 | .03 | .06 |

- Language costs about 10-15 points vs the vector (0.84-0.90 vs 1.00): the 1 mm quantisation, template variety and learning to read digits from the
  loss alone all add burden. Unlike the vector, language needs more than one episode (0.84 at h1, ~0.89 from h2): a second command helps.
- Controls are at chance for all (shuffled/no feedback/no context), so the gain comes from the command content.
- The frozen sentence encoder is much worse (0.27 -> 0.48): pooled sentence embeddings are dominated by the template wording and barely resolve the
  numbers (cosine of "4.2 cm" vs "4.3 cm" commands ~0.64, vs ~0.72 to a very different command). Pretraining did not help here; it hurts.
- Not tried: token-level T5-small features, spelled-out numbers, fewer templates, fine-tuning the encoder.

## Training curves and video
- Curves (eval success_once per DAgger iteration at h=0/1/4, all three runs): `logs/renders/icl-peg/peg_language_curves.png`
  (`plot_icl.py --hs 0 1 4`). The vector reaches ~0.95 in 5 iterations; from-scratch language needs ~25 and is still rising at 30; MiniLM is slowest and flat-ish.
- Video: `logs/renders/icl-peg/videos/peg_language.mp4` (`render_icl_language.py`): 4 tasks, h=0 on top (all fail), h=1 below with the h=0 command shown.
  Tasks are picked among those that fail at h=0 and succeed at h=1 (419/512 eval tasks do), so it shows the typical gain, not the failure cases.

## Per-token pretrained features (language_tokens)
- Per-word-piece frozen T5 features reach 0.81-0.87 (h1-h4), within noise of the from-scratch vocabulary (0.84-0.90) and far above the pooled sentence
  embedding (0.27-0.48). So the pooled vector, not the pretrained LM, was the bottleneck: per-token detail keeps the numbers. Pretraining did not
  beat the hand-built vocabulary, but it removes the need for one, so general text feedback is feasible at similar cost.
- Learning curves: T5 tokens track from-scratch tokens through iteration 15 and are still rising at 30 (h1 ~0.8), so neither is converged.
  Controls (shuffled / no feedback / no context) are at chance.
- Caveat: the commands still come from the 36-template generator; this removes the custom vocabulary, not the templates.
