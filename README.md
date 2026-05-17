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

| Branch | Description | Key Change | Dataset / Experiment | Reproduce (from `src/networks/`) | Test LSD full (L/R) | Test LSD <11k (L/R) |
|---|---|---|---|---|---|---|
| `master` | Pre-personalization baseline. Original stacked pipeline from Kestler et al. (2019). | — | cipic / trunc_64 | *(retrain needed)* | — / — | — / — |
| `personalization-1.0.0` | Main personalization branch. Adapts the stacked pipeline to generate HRTFs from anthropometrics alone. Experiments with dropout and multiple dataset variants. | Added `dropout` param | cipic / no dropout (0%) | `python main_network.py cipic all -a predict --tag Dr00-lr0005` | 5.84 / 5.49 dB | 3.98 / 4.40 dB |
| | | | cipic / dropout 1% | `python main_network.py cipic all -a predict --tag Dr001_lr0005` | 5.97 / 4.99 dB | 4.11 / 3.77 dB |
| | | | cipic / dropout 5% | `python main_network.py cipic all -a predict --tag Dr005_lr0005` | 5.39 / 5.31 dB | 3.58 / 3.54 dB |
| | | | cipic / dropout 10% | `python main_network.py cipic all -a predict --tag Dr01_lr0005` | **5.33 / 5.13 dB** | **3.86 / 4.00 dB** |
| | | | cipic / dropout 20% | `python main_network.py cipic all -a predict --tag Dr02_lr0005` | 5.50 / 5.52 dB | 3.94 / 4.11 dB |
| | | | cipic-corr-height / no dropout | `python main_network.py cipic-corr-height all -a predict --tag Dr00-lr0005` | 5.16 / 5.95 dB | 3.78 / 4.15 dB |
| | | | cipic-corr-height / dropout 10% | `python main_network.py cipic-corr-height all -a predict --tag Dr01_lr0005` | **4.94 / 5.65 dB** | **3.62 / 4.17 dB** |
| | | | Smoot_Dec_2020 / notch_smoothing_1 | `python main_network.py cipic_latest_Smoot_Dec_2020 all -t notch_smoothing_1 -a predict --tag Dr00-lr0005` | 5.86 / 5.69 dB | 3.92 / 4.24 dB |
| | | | Smoot_Dec_2020 / notch_smoothing_2 | `python main_network.py cipic_latest_Smoot_Dec_2020 all -t notch_smoothing_2 -a predict --tag Dr00-lr0005` | 5.40 / 5.80 dB | 3.89 / 4.14 dB |
| | | | Smoot_Dec_2020 / notch_smoothing_3 | `python main_network.py cipic_latest_Smoot_Dec_2020 all -t notch_smoothing_3 -a predict --tag Dr00-lr0005` | 5.51 / 5.71 dB | 3.91 / 4.33 dB |
| | | | Smoot_Dec_2020 / 5thring_6 | `python main_network.py cipic_latest_Smoot_Dec_2020 all -t 5thring_6 -a predict --tag Dr00-lr0005` | 5.41 / 5.24 dB | 3.85 / 4.15 dB |
| `personalization-1.0.1` | Removes height and seated height (x14, x15) from head anthropometric inputs. Tests whether those two CIPIC measurements add noise. | `np.delete(head_local, [13, 14])` — 15 head params instead of 17 | cipic / no dropout | `python main_network.py cipic all -a predict --tag Dr00-lr0005` | ~5.1 dB | — |
| | | | cipic / dropout 10% | `python main_network.py cipic all -a predict --tag Dr01_lr0005_removed_high` | — | — |
| `personalization-1.0.2` | LR scheduler experiment. Replaces fixed lr with `ReduceLROnPlateau` (halves lr on plateau, stops at min). lr=0.001. | `ReduceLROnPlateau` callback, `iterations=1, epochs=400` | cipic | *(retrain needed — no saved weights found)* | 5.78 / 5.03 dB | 4.25 / 3.96 dB |
| `personalization-1.1.1` | Spatial masking and weighting experiments. Rewrites `Network` as `tf.keras.Model` subclass for custom loss masking per position. | `mask_type` param, custom `train_step`, `mask_loss()` | cipic / lateral masking | `python main_network.py cipic all -a predict --tag Lateral_masked_1` | ~5.2 dB | — |
| | | | cipic / lateral weighting | `python main_network.py cipic all -a predict --tag Lateral_weighted_1` | — | — |
| | | | cipic / left-right masking | `python main_network.py cipic all -a predict --tag Left_right_masked_1` | — | — |
| | | | cipic / left-right weighting | `python main_network.py cipic all -a predict --tag Left_right_weighted_1` | — | — |
| | | | cipic / single point | `python main_network.py cipic all -a predict --tag Single_point` | — | — |
| | | | cipic / single area | `python main_network.py cipic all -a predict --tag Single_area` | — | — |
| `only_magtotal_fully_connected` | "One Model" experiment. Replaces the entire 4-stage stacked pipeline with a single fully-connected network. Same LSD as stacked pipeline but trains in ~1 hour instead of 12+. | `NetworkMagTotal.make_model()` replaced with simple Dense stack | cipic / lateral masking | `python main_network.py cipic all -a predict --tag Lateral_mask_el=0` | 5.84 / 5.86 dB | 3.65 / 4.01 dB |
| | | | cipic / single point (pos 165) | `python main_network.py cipic all -a predict --tag Center=165_t=0.2` | — | — |
| | | | cipic / single point (pos 615) | `python main_network.py cipic all -a predict --tag Center=615_t=0.2` | — | — |
| | | | cipic / single point (pos 8) | `python main_network.py cipic all -a predict --tag Center=8_t=0.2` | — | — |
| `one_point_testing` | Experimental scratchpad for single-point and single-area masking development. Predecessor to `1.1.1` and `only_magtotal_fully_connected`. | — | — | — | — | — |
