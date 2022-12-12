from data import create_dataset
from preprocess import augment, augment_changeable
import preprocess

from skimage.transform import resize
from skimage import io
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import argparse
import itertools
from distutils.util import strtobool

def generate_ablation_params():
    cases = [2, 4, 3, 1]
    size = [16]
    denoisers = ["DWT", "NL", None]
    heq = ["HE", "CLAHE", None]
    segmentation = [True, False]

    return list(itertools.product(cases, size, denoisers, heq, segmentation))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_PCA', dest='use_PCA', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--dim_PCA', type=int, default=60)
    parser.add_argument('--use_SARBake', dest='use_SARBake', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--case', type=int, help='case', default=2)
    args = parser.parse_args()

    aug_args = [16, "DWT", "HE", False]
    #aug_args = generate_ablation_params()[i][1:]

    use_PCA = args.use_PCA
    dim_PCA = args.dim_PCA
    use_SARBake = args.use_SARBake
    case = args.case

    if use_SARBake:
        train_ds_path = "SARBake/TRAIN_17"
        test_ds_path = "SARBake/TEST_15"
        case = 5
    else:
        train_ds_path = "dataset/TRAIN_17"
        test_ds_path = "dataset/TEST_15"

    if use_PCA:
        pca = PCA(n_components=dim_PCA, svd_solver="arpack") # Set SVD solver, otherwise it will not be deterministic!
        train_imgs, train_labels = create_dataset(train_ds_path, case=case, augment=augment_changeable, augment_args=aug_args)
        print("Created training dataset.")
        
        pca.fit(train_imgs)
        reduced_train_imgs = []
        for i in range(len(train_imgs)):
            x = train_imgs[i].reshape(1,-1)
            x = pca.transform(x)
            x = x.flatten()
            reduced_train_imgs.append(x)

        test_imgs, test_labels = create_dataset(test_ds_path, case=case, augment=augment_changeable, augment_args=aug_args)
        print("Created test dataset.")

        reduced_test_imgs = []
        for i in range(len(test_imgs)):
            x = test_imgs[i].reshape(1,-1)
            x = pca.transform(x)
            x = x.flatten()
            reduced_test_imgs.append(x)

        svc = SVC()
        svc.fit(reduced_train_imgs, train_labels)
        pred_labels = svc.predict(reduced_test_imgs)
        acc = accuracy_score(pred_labels, test_labels)
    else:
        train_imgs, train_labels = create_dataset(train_ds_path, case=case, augment=augment_changeable, augment_args=aug_args)
        print("Created training dataset.")
        
        test_imgs, test_labels = create_dataset(test_ds_path, case=case, augment=augment_changeable, augment_args=aug_args)
        print("Created test dataset.")

        svc = SVC()
        svc.fit(train_imgs, train_labels)
        pred_labels = svc.predict(test_imgs)
        acc = accuracy_score(pred_labels, test_labels)