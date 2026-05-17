# HRTF Machine Learning Project

Generate an HRTF based on the input of direction of source, ear size, and headsize


## HDF5 Format

Datasets are stored in the hdf5 format with the following structure
	
	subject
		hrir_l, hrir_r, srcpos
		attrs
			fs
			(depends on dataset)

Additionaly spherical coordinates are represented in the folloing manner 

```math
(azimuth\ \theta, elevation\ \phi, r)
```
with the following conversions
```math
x = r\cos(\theta)\cos(\phi)
```
```math
y = r\sin(\theta)\cos(\phi)
```
```math
z = r\sin(\phi)
```
(0, 90) is directory above. (0, 0) is directly in front. (90, 0) is directory left. (90,0) is directly right.


## Cipic Docs

| Var	        |		Meaning				|
|---------------|---------------------------|
|$`x_{1}`$		| head width				|
|$`x_{2}`$		| head height				|
|$`x_{3}`$		| head depth				|
|$`x_{4}`$		| pinna offset down			|
|$`x_{5}`$		| pinna offset back			|
|$`x_{6}`$		| neck width				|
|$`x_{7}`$		| neck height				|
|$`x_{8}`$		| neck depth				|
|$`x_{9}`$		| torso top width			|
|$`x_{10}`$		| torso top height			|
|$`x_{11}`$		| torso top depth			|
|$`x_{12}`$		| shoulder width			|
|$`x_{13}`$		| head offset forward			|
|$`x_{14}`$		| height					|
|$`x_{15}`$		| seated height				|
|$`x_{16}`$		| head circumference		|
|$`x_{17}`$		| shoulder circumference	|
|$`d_{1}`$		| cavum concha height		|
|$`d_{2}`$		| cymba concha height		|
|$`d_{3}`$		| cavum concha width		|
|$`d_{4}`$		| fossa height				|
|$`d_{5}`$		| pinna height				|
|$`d_{6}`$		| pinna width				|
|$`d_{7}`$		| intertragal incisure width|
|$`d_{8}`$		| cavum concha depth		|
|$`\theta_{1}`$	| pinna rotation angle		|
|$`\theta_{2}`$	| pinna flare angle			|

<img src="HeadMeasurements.png">

<img src="PinnaMeasurements.png">


## Setup

```bash
conda create -n hrtf python=3.9.16
conda activate hrtf
pip install -r requirements.txt
```

## Branch Map

### Running training and prediction

All commands run from `src/networks/`. Train:

```bash
cd src/networks
python main_network.py cipic all -a train --tag <tag>
```

Predict (loads saved weights, enters interactive mode):

```bash
cd src/networks
python main_network.py cipic all -a predict --tag <tag>
```

At the interactive prompt:
```
lsd 0    # LSD for CIPIC subject 003 (training) — read the "Left/Right magtotal [full, <11k]" line
lsd 2    # LSD for CIPIC subject 009 (test)      — read the "Left/Right magtotal [full, <11k]" line
```

> **Note:** `lsd N` index is not the CIPIC subject ID. CIPIC skips many IDs (000–002, 004–007, … don't exist), so the dataset stacks only valid subjects in order. Index 0 = subject 003, index 2 = subject 009.

### Experiments

LSD reported for CIPIC subject 003 (training) and subject 009 (test). "Full" = full spectrum, "<11k" = below 11 kHz.

| Branch | Description | Key Change | Reproduce (from `src/networks/`) | Subj: 009 Test LSD — (L / R) | Test LSD — <11kHz (L / R) |
|---|---|---|---|---|---|
| `master` | <div style="width:300px">Pre-personalization baseline. Original stacked pipeline from Kestler et al. (2019).</div> | — | —  | —  / —  | —  / —  |
| `personalization-1.0.0` | <div style="width:300px">**Main personalization branch.** Adapts the stacked pipeline to generate HRTFs from anthropometrics alone. Experiments with dropout (0/1/5/10/20%) and cleaned/smoothed CIPIC dataset variants (notch-smoothing-1/2/3, 5thring-6, cipic-corr-height).</div> | <div style="width:300px">Added `dropout` param; multiple dataset variants</div> | **[cipic] no dropout (0%):** `python main_network.py cipic all -a predict --tag Dr00-lr0005`<br>**[cipic] dropout 1%:** `python main_network.py cipic all -a predict --tag Dr001_lr0005`<br>**[cipic] dropout 5%:** `python main_network.py cipic all -a predict --tag Dr005_lr0005`<br>**[cipic] dropout 10%:** `python main_network.py cipic all -a predict --tag Dr01_lr0005`<br>**[cipic] dropout 20%:** `python main_network.py cipic all -a predict --tag Dr02_lr0005`<br><hr>**[cipic-corr-height] no dropout:** `python main_network.py cipic-corr-height all -a predict --tag Dr00-lr0005`<br>**[cipic-corr-height] dropout 10%:** `python main_network.py cipic-corr-height all -a predict --tag Dr01_lr0005`<br><hr>**[cipic_latest_Smoot_Dec_2020] notch_smoothing_1 (no dropout):** `python main_network.py cipic_latest_Smoot_Dec_2020 all -t notch_smoothing_1 -a predict --tag Dr00-lr0005`<br>**[cipic_latest_Smoot_Dec_2020] notch_smoothing_2 (no dropout):** `python main_network.py cipic_latest_Smoot_Dec_2020 all -t notch_smoothing_2 -a predict --tag Dr00-lr0005`<br>**[cipic_latest_Smoot_Dec_2020] notch_smoothing_3 (no dropout):** `python main_network.py cipic_latest_Smoot_Dec_2020 all -t notch_smoothing_3 -a predict --tag Dr00-lr0005`<br>**[cipic_latest_Smoot_Dec_2020] 5thring_6 (no dropout):** `python main_network.py cipic_latest_Smoot_Dec_2020 all -t 5thring_6 -a predict --tag Dr00-lr0005` | 5.84 / 5.49 dB<br>5.97 / 4.99 dB<br>5.39 / 5.31 dB<br>**5.33 / 5.13 dB**<br>5.50 / 5.52 dB<br><hr>5.16 / 5.95 dB<br>**4.94 / 5.65 dB**<br><hr>5.86 / 5.69 dB<br><br>5.40 / 5.80 dB<br><br>5.51 / 5.71 dB<br><br>5.41 / 5.24 dB<br><br> | 3.98 / 4.40 dB<br>4.11 / 3.77 dB<br>3.58 / 3.54 dB<br>**3.86 / 4.00 dB**<br>3.94 / 4.11 dB<br><hr>3.78 / 4.15 dB<br>**3.62 / 4.17 dB**<br><hr>3.92 / 4.24 dB<br><br>3.89 / 4.14 dB<br><br>3.91 / 4.33 dB<br><br>3.85 / 4.15 dB<br><br> |
| `personalization-1.0.1` | <div style="width:300px">Removes height and seated height (x14, x15) from head anthropometric inputs. Tests whether those two CIPIC measurements add noise.</div> | <div style="width:300px">`np.delete(head_local, [13, 14])` — 15 head params instead of 17</div> | **no dropout:** `python main_network.py cipic all -a predict --tag Dr00-lr0005`<br>**dropout 10%:** `python main_network.py cipic all -a predict --tag Dr01_lr0005_removed_high` | ~5.1 dB<br>— | —<br>— |
| `personalization-1.0.2` | <div style="width:300px">LR scheduler experiment. Replaces fixed lr with `ReduceLROnPlateau` (halves lr on plateau, stops at min). lr=0.001.</div> | <div style="width:300px">`ReduceLROnPlateau` callback, `iterations=1, epochs=400`</div> | *(retrain needed — no saved weights found)* | 5.78 / 5.03 dB | 4.25 / 3.96 dB |
| `personalization-1.1.1` | <div style="width:300px">Spatial masking and weighting experiments. Rewrites `Network` as `tf.keras.Model` subclass for custom loss masking per position. Implements: lateral masking, lateral weighting, left-right masking/weighting, single-point masking, single-area masking.</div> | <div style="width:300px">`mask_type` param, custom `train_step`, `mask_loss()`</div> | **lateral masking:** `python main_network.py cipic all -a predict --tag Lateral_masked_1`<br>**lateral weighting:** `python main_network.py cipic all -a predict --tag Lateral_weighted_1`<br>**left-right masking:** `python main_network.py cipic all -a predict --tag Left_right_masked_1`<br>**left-right weighting:** `python main_network.py cipic all -a predict --tag Left_right_weighted_1`<br>**single point:** `python main_network.py cipic all -a predict --tag Single_point`<br>**single area:** `python main_network.py cipic all -a predict --tag Single_area` | ~5.2 dB<br>—<br>—<br>—<br>—<br>— | —<br>—<br>—<br>—<br>—<br>— |
| `only_magtotal_fully_connected` | <div style="width:300px">"One Model" experiment. Replaces the entire 4-stage stacked pipeline with a single fully-connected network. Same LSD as stacked pipeline but trains in ~1 hour instead of 12+. Also includes single-point and single-area experiments from `1.1.1`.</div> | <div style="width:300px">`NetworkMagTotal.make_model()` replaced with simple Dense stack</div> | **lateral masking:** `python main_network.py cipic all -a predict --tag Lateral_mask_el=0`<br>**single point (pos 165):** `python main_network.py cipic all -a predict --tag Center=165_t=0.2`<br>**single point (pos 615):** `python main_network.py cipic all -a predict --tag Center=615_t=0.2`<br>**single point (pos 8):** `python main_network.py cipic all -a predict --tag Center=8_t=0.2` | 5.84 / 5.86 dB<br>—<br>—<br>— | 3.65 / 4.01 dB<br>—<br>—<br>— |
| `one_point_testing` | <div style="width:300px">Experimental scratchpad for single-point and single-area masking development. Predecessor to the masking work in `1.1.1` and `only_magtotal_fully_connected`.</div> | <div style="width:300px">—</div> | — | — | — |
