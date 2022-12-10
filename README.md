# MSTAR10_classify
Bilkent University CS554 Computer Vision term project. SAR object classification with classical methods on MSTAR-10 class dataset.

## Setup

### Manual dataset creation
* **Public SDMS MSTAR dataset** can be downloaded from [this](https://www.sdms.afrl.af.mil/index.php?collection=public-data&page=public-data-list) link. 

  * There is not an already prepared 10-class dataset of MSTAR. You should download **MSTAR Clutter** and **MSTAR Target Chips (T72 BMP2 BTR70 SLICY)**, extract 10 classes *(2S1, BMP2, BDRM_2, BTR70, D7, T62, T72, ZIL131, ZSU_23_4, BTR_60)* according to their test (15 degree azimuth angle) and train (17 degree azimuth angle) sets.

  * After organizing the 10-class RAW files, extract JPEG, HDR and MAG files using the [MSTAR PUBLIC TOOLS](https://www.sdms.afrl.af.mil/index.php?collection=tools_mstar). You may modify the provided `generate_hdr_mag.bash` and use for batch processing.

* For masked ground-truth data, we use **SARBake**[^fn1]. You may use the utility scripts in `data.py` to convert the masks from CSV to PNG.


### Pre-processed dataset
Since above method is cumbersome very time consuming, we [provide the organized and ready-to-use MSTAR dataset](https://drive.google.com/file/d/1G_EbJug8qexX48VMUzFlKyjz81IsZAWo/view?usp=share_link). Each data in this dataset has 7 variations:
* RAW
* HDR
* MAG
* JPG/JPEG
* Oriented images
* Oriented + CLEANed images
* CLEANed images

We also [provide](https://drive.google.com/file/d/1u3pAQp5DBjdHTn-8l8pwgJaLXOu7frHU/view?usp=share_link) the PNG masks of SARBake. 

## Methodology

## Run



[^fn1]: So Chris Krycho, "Not Exactly a Millennium," chriskrycho.com, July 22,
