# MSTAR10_classify
SAR object classification with classical methods on MSTAR-10 class dataset.

## Setup

### Environment 
```
python 3.7.10
numpy==1.19.1
opencv-python==4.5.1
Pillow==9.3.0
PyWavelets==1.1.1
scikit_image==0.17.2
scikit_learn==1.2.0
scipy==1.5.2
skimage==0.0
```
### Manual dataset creation
**Public SDMS MSTAR dataset** can be downloaded from [this](https://www.sdms.afrl.af.mil/index.php?collection=public-data&page=public-data-list) link. 
* There is not an already prepared 10-class dataset of MSTAR. You need to download **MSTAR Clutter** and **MSTAR Target Chips (T72 BMP2 BTR70 SLICY)**, extract 10 classes *(2S1, BMP2, BDRM_2, BTR70, D7, T62, T72, ZIL131, ZSU_23_4, BTR_60)* according to their test (15 degree azimuth angle) and train (17 degree azimuth angle) sets.

* After organizing the 10-class RAW files, extract JPEG, HDR and MAG files using the [MSTAR PUBLIC TOOLS](https://www.sdms.afrl.af.mil/index.php?collection=tools_mstar). You may modify the provided `generate_hdr_mag.bash` and use for batch processing.

For masked ground-truth data, we use [**SARBake**](https://data.mendeley.com/datasets/jxhsg8tj7g/3)[^fn1]. You may use the utility scripts in `data.py` to convert the masks from CSV to PNG, and you need to manually seperate 15-17 degrees of azimuth angles for each class since they are combined.


### Pre-processed dataset
We [provide the organized, pre-processed, and ready-to-use MSTAR dataset](https://drive.google.com/file/d/1G_EbJug8qexX48VMUzFlKyjz81IsZAWo/view?usp=share_link) to avoid all the hassle. Each data in this dataset has 7 variations:
* RAW
* HDR
* MAG
* JPG/JPEG
* Oriented images
* Oriented + CLEANed images
* CLEANed[^fn2] images

We also [provide the PNG target masks of SARBake](https://drive.google.com/file/d/1u3pAQp5DBjdHTn-8l8pwgJaLXOu7frHU/view?usp=share_link). Each data in this dataset has 3 variations:
* PNG masks
* CSV masks
* PNG of the original SAR image + mask boundaries

## Methodology

* We follow a pipelined structure for the MSTAR 10-class classification task, and experiment with various methods and their combinations to get the highest classification accuracy.

1. **Preprocessing:** Histogram equalization, CLAHE, non-local means denoising, DWT denoising, morphological operations, image rotation, image resizing
2. **Feature extraction:** CLEAN, Gabor filters, PCA
3. **Classification:** SVM

* For CLEAN, a custom point spread function $h$ is implemented for SAR ASC[^fn3], using the bandwidth $B$, center frequency $f_c$, azimuth angle $\Theta$ and center azimuth $\theta_c$ from HDR files.:

$$ CLEAN(I_{noisy}(x,y), h) = I_{clean}(x,y) $$ 

$$ I_{clean}(x,y) = \Sigma_{n=1}^{N}A_N h(x-x_n, y-y_n) $$

$$ h(x,y) = e^{\frac{j 4 \pi f_c}{c}{(x+\theta_c y)}}{\frac{4 f_c B \Theta}{c^2}} sinc(\frac{2B}{c}x) sinc(\frac{2 f_c \Theta}{c}y) \omega(x,y) $$

$\omega$ is the -35dB Taylor window function that was originally utilized while collecting the MSTAR data.

* We observe that downscaling to 16x16 + setting all azimuth angles to 0 (image rotation) + histogram equalization + DWT denoising + 60-dim PCA features + SVM combination yields the highest accuracy (98.9%) among all tried combinations.

## Run

* Please download the two pre-processed datasets and extract to the project root folder. Make sure that the project folder directory tree looks as follows:
```
.
├── data.py
├── main.py
├── preprocess.py
├── dataset
│   ├── Same directory layout as SARBake
└── SARBake
    ├── TEST_15
    │   ├── 2S1
    │   ├── BMP2
    │   │   ├── SN_9563
    │   │   ├── SN_9566
    │   │   └── SN_C21
    │   ├── BRDM_2
    │   ├── BTR_60
    │   ├── BTR70
    │   │   └── SN_C71
    │   ├── D7
    │   ├── T62
    │   ├── T72
    │   │   ├── SN_132
    │   │   ├── SN_812
    │   │   └── SN_S7
    │   ├── ZIL131
    │   └── ZSU_23_4
    └── TRAIN_17
        └── Same layout as TEST_15
```
* `python3 main.py --use_PCA=true --use_SARBake=false --case=2`

* `case` handles which files to be read:
    * `case=0` : Read hdr files
    * `case=1` : Read CLEANed images
    * `case=2` : Read oriented images
    * `case=3` : Read oriented+CLEANed images
    * `case=4` : Read vanilla MSTAR images
    * `case=5` : Read png-converted csv masks of SARBake overlays

[^fn1]: Malmgren-Hansen, David; Nobel-Jørgensen, Morten (2017), “SARBake Overlays for the MSTAR Dataset”, Mendeley Data, V3, doi: 10.17632/jxhsg8tj7g.3
[^fn2]: https://github.com/vit1-irk/clean_lib
[^fn3]: L. C. Potter and R. L. Moses, "Attributed scattering centers for SAR ATR," in IEEE Transactions on Image Processing, vol. 6, no. 1, pp. 79-91, Jan. 1997, doi: 10.1109/83.552098.
