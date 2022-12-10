import numpy as np
from numpy import genfromtxt
import cv2
import os
import linecache
from scipy import ndimage as nd
from skimage import io
from sklearn.utils import shuffle

import preprocess

def read_azimuth_angles(header_path):
    """
    Returns the azimuth angle from the `header_path`, with extension *.hdr.
    """
    try: # Header files have the azimuth angles in 16th or 17th lines
        line = linecache.getline(header_path, 16)
        attribute = str(line.split('=')[-2])
        assert attribute == "TargetAz"
    except:
        line = linecache.getline(header_path, 17)
        attribute = str(line.split('=')[-2])
        assert attribute == "TargetAz"

    angle = float(line.split('=')[-1])
    return angle

def generate_CLEAN_images(header_path, clean_oriented_imgs=True, write_imgs=True, apply_preprocess=True):
    """
    Extracts azimuth angle from `header_path`, generates PSF with the extracted parameters, 
    applies CLEAN deconvolution algorithm, and writes the CLEANed images to the same path as `header_path`.
    """
    if not clean_oriented_imgs: # CLEAN non-oriented images, get azimuth angle
        angle = read_azimuth_angles(header_path)
    else: # Target azimuth angle is oriented already, CLEAN oriented images
        angle = 2*np.pi

    try: # Classes with JPEG images have an additional sub-folder, JPG ones do not
        img_path = '.'.join(header_path.split('.')[:-1])+'.jpeg'
    except:
        img_path = '.'.join(header_path.split('.')[:-2])+'.JPG'
    
   
    splitted_path = img_path.split('.')
    target_path = ('.'.join(splitted_path[:-1])+'.CLEANed.'+splitted_path[-1])
    
    img = io.imread(img_path)
    if apply_preprocess:
        img = preprocess.histogram_equalization(img)
    print(f"Read image: {img_path}")

    # CLEAN process
    fc = 9.599000*1e9 #ghz->hz, center frequency
    B = 0.591*1e9 #ghz->hz, bandwidth
    psf_real = preprocess.create_ASC_PSF(img=img, fc=fc, B=B, az=angle)
    clean = preprocess.run_CLEAN(img=img, PSF=psf_real, blurrad_w1=2, gamma_w1=0.6, bottomlimit=0.01)

    print(f"Cleaned: {img_path}")
    if write_imgs:
        cv2.imwrite(target_path, clean)

def generate_mask_from_csv(csv_mask_path, tag="target", remove_shadow=True):
    """ 
    Generates png masks from SARBake overlays:
    
    Malmgren-Hansen, David; Nobel-Jørgensen, Morten (2017), “SARBake Overlays for the MSTAR Dataset”, Mendeley Data, V3, doi: 10.17632/jxhsg8tj7g.3
    https://data.mendeley.com/datasets/jxhsg8tj7g/3

    Values in the CSV files of the overlays:
    0 = background,
    1 = target,
    2 = shadow
    """
    mask = genfromtxt(csv_mask_path, delimiter=',')
    mask[mask>1] = 0 if remove_shadow else 0.5
    mask_name = csv_mask_path[:-3]+tag+'.png'
    cv2.imwrite(mask_name, 255*mask)
    print(f'Generated png mask:{mask_name}')

def orient_imgs(header_path):
    """
    Get target azimuth angles from *.hdr `header_path`, orient with 0 degree angle, rotate and save the new image.\n
    *.hdr files can be generated via `generate_hdr_mag.bash` script, which uses official MSTAR utility C scripts.\n
    Source images (*.jpeg/*.jpg) have to be in the same folder with the *.hdr files.
    """
    try: # Header files have the azimuth angles in 16th or 17th lines
        line = linecache.getline(header_path, 16)
        attribute = str(line.split('=')[-2])
        assert attribute == "TargetAz"
    except:
        line = linecache.getline(header_path, 17)
        attribute = str(line.split('=')[-2])
        assert attribute == "TargetAz"
    
    angle = float(line.split('=')[-1])

    try: # Classes with JPEG images have an additional sub-folder, JPG ones do not
        img_path = '.'.join(header_path.split('.')[:-1])+'.jpeg'
    except:
        img_path = '.'.join(header_path.split('.')[:-2])+'.JPG'
    
    splitted_path = img_path.split('.')
    target_path = ('.'.join(splitted_path[:-1])+'.ornt.'+splitted_path[-1])
    print(target_path)

    img = io.imread(img_path)
    img_rotated = nd.rotate(img, angle, reshape=False)
    cv2.imwrite(target_path, img_rotated)

    linecache.clearcache()

def process_dataset(ds_folder, extension, transform):
    """
    Given the `transform` and `ds_folder`, applies the transform to the files 
    in all subfolders, with specific `extension`s in the dataset.
    ----------
    For instance:

    modify_dataset(ds_folder, 'hdr', orient_imgs)

    modify_dataset(ds_folder, 'hdr', generate_CLEAN_images)
    
    modify_dataset(ds_folder, 'csv', generate_mask_from_csv)
    """
    for path, _, files in os.walk(ds_folder):
        for file in files:
            ext = file.split('.')[-1]
            if ext == extension:
                transform(os.path.join(path, file))

def create_dataset(ds_folder, augment, case=4, shuffle_ds=True):
    """
    Creates the a dataset (images, labels) iterating all subfolders in MSTAR `ds_folder`.
    
    `case` handles which files to be read from `ds_folder` and sent to `augment`. 
        `case=0` : Read hdr files
        `case=1` : Read CLEANed images
        `case=2` : Read oriented images
        `case=3` : Read oriented+CLEANed images
        `case=4` : Read vanilla MSTAR images
        `case=5` : Read png-converted csv masks of SARBake overlays

    `ds_folder` must include the files specified with `case`, change accordingly.
        
    ----------
    Dataset structure is as follows:

    if `ds_folder` has MSTAR images:
        Each can have 7 extensions: [RAW, HDR, MAG, JPG/JPEG, CLEANed.jpg/jpeg, ornt.CLEANed.jpg/jpeg]\n
        Subfoldered classses (BMP2, BTR70, T72) have *.jpeg images whereas the others have *jpg.
    if `ds_folder` has SARBake overlays:
        Each can have 3 extensions: [png, target.png, CSV]
    
    dataset:
        TRAIN_17
            2S1: 299
            BMP2:
                SN_9563: 233
                SN_9566: 232
                SN_C21: 233
            BRDM_2: 298
            BTR70:
                SN_C71: 233
            BTR_60: 256
            D7: 299
            T62: 299
            T72:
                SN_132: 232
                SN_812: 231
                SN_S7: 228
            ZIL131: 299
            ZSU_23_4: 299

        TEST_15
            2S1: 274
            BMP2:
                SN_9563: 195
                SN_9566: 196
                SN_C21: 196
            BRDM_2: 274
            BTR70: 
                SN_C71: 196
            BTR_60: 195
            D7: 274
            T62: 273
            T72:
                SN_132: 196
                SN_812: 195
                SN_S7: 191
            ZIL131: 274
            ZSU_23_4: 274
    """

    def extract_labels():
        # Extract labels from sub-folder and folder names
        splitted_path = path.split("/")
        if ext == 'JPG': # Modified MSTAR dataset
            label = splitted_path[-1]
        elif ext == 'jpeg':
            label = splitted_path[-2]
        elif ext == 'png': # SARBake overlays
            if splitted_path[-3] == 'TEST_15' or splitted_path[-3] == 'TRAIN_17':
                label = splitted_path[-2]
            else:
                label = splitted_path[-1]
        #print(f'Read:{file}, class:{label}')
        return label

    features_list = []
    labels = []
    classes = ["2S1", "BMP2", "BRDM_2", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU_23_4", "BTR_60"]
    
    for path, _, files in os.walk(ds_folder):
        for file in files:
            ext = file.split('.')[-1]
            ext_with_flag = file.split('.')[-3:-1]
            if ext == 'hdr':
                if case == 0:
                    augment(os.path.join(path, file))
                    extract_labels()
                    continue
            elif ext == 'jpeg' or ext == 'JPG':
                # Only CLEANed images
                if ext_with_flag[-1] == 'CLEANed' and ext_with_flag[-2] != 'ornt': 
                    if case == 1:
                        features = augment(os.path.join(path, file))
                        features_list.append(features)
                        label = extract_labels()
                        labels.append(classes.index(label))
                    continue
                # Only oriented images
                elif ext_with_flag[-1] == 'ornt': 
                    if case == 2:
                        features = augment(os.path.join(path, file))
                        features_list.append(features)
                        label = extract_labels()
                        labels.append(classes.index(label))
                    continue
                # Both oriented and CLEANed images
                elif '.'.join(ext_with_flag) == 'ornt.CLEANed': 
                    if case == 3:
                        features = augment(os.path.join(path, file))
                        features_list.append(features)
                        label = extract_labels()
                        labels.append(classes.index(label))
                    continue
                # Original MSTAR images
                else:  
                    if case == 4:
                        features = augment(os.path.join(path, file))
                        features_list.append(features)
                        label = extract_labels()
                        labels.append(classes.index(label))
                    continue
            # SARBake overlays
            elif ext == 'png': 
                # *.png versions of the *.csv segmentation masks (outputs of generate_mask_from_csv function)
                if ext_with_flag[-1] == 'target':
                    if case == 5:
                        features = augment(os.path.join(path, file))
                        features_list.append(features)
                        label = extract_labels()
                        labels.append(classes.index(label))
                    continue
    if shuffle_ds:
        features_list, labels = shuffle(features_list, labels)  # unison shuffling

    return features_list, labels

if __name__=='__main__':
    process_dataset("dataset/TRAIN_17", extension='hdr', transform=generate_CLEAN_images)
    #modify_dataset("/home/batu/Desktop/554_PROJECT/SARBake", generate_mask_from_csv)