# The SOG camera block (mono splats)

What the splat capability writes into a `.sog` so a viewer knows the camera the Gaussians were
lifted through, and the one rule that decides its focus. This block must match the DisplayXR
Gallery's online SOG generator in meaning, field for field and rule for rule: a second, subtly
different definition of the focus shows up only as the subject drifting under head motion.
**Any change here must be made in both.**

Code: `src/oc_sog.c` (container, planes, block text), `src/oc_splat.c`
(`oc_splat_nearest_clump`). Tests: `tests/sog_test.c`.

## 1. Container

- **PlayCanvas SOG version 2.** A ZIP whose members are `means_l.webp`, `means_u.webp`,
  `quats.webp`, `scales.webp`, `sh0.webp` (lossless WebP planes) and `meta.json`, every member
  **stored** (no compression, no data descriptors, no ZIP64). Degree 0: no `shN`.
- `meta.json` is splat-transform 1.10's version-2 document (`version`, `asset`, `count`, `means`,
  `scales`, `quats`, `sh0`) plus one top-level key, `camera`, **immediately after `count`**,
  serialised on **one line** (compact separators). Readers ignore unknown top-level keys.
- `meta.json` stays at or below **65,536 bytes** (SOG readers refuse larger; this one is about
  7.5 KB).
- **Why version 2:** the native DisplayXR Gaussian Splat viewer reads version 2 and refuses
  version 3 ("unsupported SOG version 3"). PlayCanvas splat-transform 3.x writes version 3 by
  default; 1.10.x writes version 2. splat-transform also **drops unknown keys** when it re-encodes
  a SOG, so a transcode strips the block: write the block into the file you ship, last.
- Plane encodings (one texel per Gaussian, Morton order; width = ceil(√N / 4)·4, height =
  ceil(N / width / 4)·4): see the header comment of `src/oc_sog.h`.

## 2. The mono block

```json
"camera":{"convention":"opencv","rig":"camera","rest":{"position":[0,0,0],"rotation":[0,0,0,1]},"intrinsics":{"fx":F,"fy":F,"cx":W/2,"cy":H/2,"width":W,"height":H},"focus":{"point":[0,0,Z],"subject_m":Z,"near_m":N,"far_m":R,"source":"nearest-clump"},"dxr":{"ipd_factor":1.0,"parallax_factor":1.0}}
```

| Field | Meaning | Units |
|---|---|---|
| `convention` | `"opencv"`: +x right, +y down, +z forward. A viewer ignores the block otherwise. | — |
| `rig` | `"camera"`: the splat was lifted from a photograph, so the eyes move about the capture camera and its field of view is what to conserve. (`"display"`, eyes about a fixed screen, is never written here.) | — |
| `rest.position` | The capture camera's position: the splat's origin **is** the capture camera, so identity. | metres |
| `rest.rotation` | Camera → world, `x y z w`: identity. | quaternion |
| `intrinsics.fx`, `fy` | The focal the Gaussians were **actually unprojected through**, in pixels of the **input image** (not the model's 1536 grid, not what was asked for): a splat built at f_s and viewed at f_v is drawn scaled by f_v / f_s. The facts' `intrinsics.source` says whether it was given or assumed (the model's 30 mm full-frame equivalent: diag_px · 30 / 43.267). | px |
| `intrinsics.cx`, `cy` | The principal point: `W/2`, `H/2` (a mono photo has no deconvergence shift; in general `cx = W/2 − infinity_shift_px`). | px |
| `intrinsics.width`, `height` | The image the model saw (after any host downscale), integers. | px |
| `focus.point` | On the optical axis at the nearest-clump distance, `[0, 0, Z]` (section 3). | metres (splat units) |
| `focus.subject_m` | The same Z. | metres |
| `focus.near_m`, `far_m` | Opacity-weighted 5th / 95th percentile of z over the same central crop; each only when > 0. | metres |
| `focus.source` | `"nearest-clump"`. A viewer logs it (`block (nearest-clump)`); it does not act on it. | — |
| `dxr.ipd_factor`, `parallax_factor` | Neutral presentation, written explicitly as `1.0` so a reader can tell "1" from "no opinion". | — |

- **No `stereo` key** for a mono splat: a guessed 63 mm baseline is exactly the error this block
  exists to prevent. No `metric` key: a monocular model's scale is not measured.
- **No clump → no `focus` key** at all (a viewer then estimates its own).
- Numbers are written in their shortest round-trip form (`1920.0`, `1.847`); integers stay
  integers.
- The job's facts (`focus_suggestion`, `intrinsics`) carry the same numbers as the block.

## 3. `nearest_clump_focus`

On the **final** Gaussians in camera space (after unprojection), with the activated opacity
(the sigmoid of the stored logit):

1. **Sample:** keep every Gaussian whose index satisfies `index % ceil(n / 40000) == 0` (at most
   40,000).
2. **Drop** opacity < **0.05** and z ≤ **0.05 m**.
3. **Central half:** keep |u| ≤ **0.25** and |v| ≤ **0.25**, where
   u = (F·x/z + cx) / W − 0.5 and v = (F·y/z + cy) / H − 0.5.
4. **Histogram:** opacity-weighted histogram of 1/z, **120** bins over [min, max] of 1/z
   (bin = min(119, floor((1/z − min) / (max − min) · 120))).
5. **Smooth:** 3-bin moving average, edge-clamped. A bin is *empty* when its smoothed mass is
   ≤ **0.1 %** of the total.
6. **Scan from the near end** (highest 1/z) for the first run of non-empty bins whose **raw**
   mass is ≥ **3 %** of the total; skip runs below that and continue past them.
7. **Z** = 1 / (opacity-weighted mean of 1/z over the Gaussians in that run), clamped to
   **[0.1, 20] m**.
8. **near_m / far_m** = opacity-weighted p05 / p95 of z over the crop of step 3 (stable sort by
   z; the first z whose cumulative weight reaches p · total).
9. No crop, no spread of 1/z, or no run reaching 3 % → **no focus**.

**Why not the median.** The opacity-weighted cloud median lands wherever most of the mass is,
and on a landscape most of the mass is the far background: the median can sit "at infinity",
and converging there flattens the subject into the screen. The nearest substantial clump in the
central half of the frame lands on the subject; the 3 % floor keeps a few stray near Gaussians
from being taken for it.

## 4. How a viewer renders it

- **Pivot D = `focus.point[2]`** is three things at once: the orbit centre, the zero-disparity
  plane (what stays put under head motion) and the vertex of the comfort cone.
- The **window** is the photo's own frustum (`fx`, `cx`, `cy`, `width`, `height`) cut at D.
- The **eyes** sit about the capture camera with identity rotation and off-axis frusta through
  that window, at the real interpupillary distance (63 mm in scene metres, times `ipd_factor`
  = 1; `parallax_factor` = 1), with no extra depth gain.
- **Check:** the focal a viewer logs (the native viewer: `Camera rig: fx=… pivot=…m (block
  (nearest-clump))`) must equal the block's `fx`. A viewer that recovers or guesses its own
  focal draws the cloud scaled by f_viewer / f_block, and the depth reads too strong (or too
  weak) for it.

## 5. The stereo / metric variant (not produced by this provider today)

A splat lifted from a stereo pair, or a mono splat fitted to its pair's disparity, is metric at
the pair's baseline: its block adds `"stereo":{"baseline_m":B}` (and a `metric` object
describing the fit), and its focus comes from the pair's convergence rather than from the
cloud: D = clamp(min(d_conv, d_subject), 0.1, 20) with d = f·B / max(disparity_px, 0.5), where
`intrinsics.cx` may carry a deconvergence shift. This provider writes mono blocks only.

## 6. Compatibility

The block above is the DisplayXR Gallery's online SOG generator's mono block, and
`nearest_clump_focus` is its rule with its constants. The two must agree byte-for-byte in
meaning: same keys, same defaults, same rule, same constants. Change both together, never one.
