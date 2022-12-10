import numpy as np
import cv2
from skimage.filters import gabor_kernel
from skimage.transform import resize
from scipy import ndimage as nd
import pywt
from skimage import io
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy import signal

def adaptive_histogram_equalization(im):
    im = (im*255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    adaptive = clahe.apply(im)
    return adaptive

def histogram_equalization(im):
    im = (im*255).astype(np.uint8)
    equalized = cv2.equalizeHist(im)
    return equalized

def nl_means_denoising(im):
    im = (im * 255).astype(np.uint8)
    im = cv2.fastNlMeansDenoising(im)
    return im

def gabor_filter_construct():
    filters = []
    for theta in range(9):
        theta = theta / 18. * np.pi
        for sigma in (1,5):
            filter = np.real(gabor_kernel(0.05, theta = theta, sigma_x = sigma,sigma_y = sigma))
            filters.append(filter)
    return filters

def gabor_filter_features(im,filters):
    features = np.zeros((len(filters),2), dtype = np.double)
    for i in range(len(filters)):
        filtered = nd.convolve(im,filters[i],mode ='wrap')
        features[i] = filtered.mean()
        features[i, 1] = filtered.var()
    return features

def wavelet_transform(im):
    coeffs_2 = pywt.dwt2(im, 'bior6.8') # there are several types of wavelets, should be examined more
    LL, (LH, HL, HH) = coeffs_2
    #lowFreq = pywt.idwt2((LL, (None, None, None)), 'haar')
    return LL, (LH, HL, HH)

def dwt_tresholding(coeffs, sigma_d=2, k=30, kind='soft',
            sigma_levels=[0.889, 0.2, 0.086, 0.041, 0.020, 0.010, 0.005, 0.0025, 0.0012], level=0):
    for idx in range(1, len(coeffs[1:])):
         coeffs[idx] = pywt.threshold(coeffs[idx], sigma_d*k*sigma_levels[level], kind)
    return coeffs[0], (coeffs[1], coeffs[2], coeffs[3])

def dwt_denoising(im):
    LL, (LH, HL, HH) = pywt.dwt2(im,'haar')
    coeffs = dwt_tresholding([LL, LH, HL, HH])
    denoised = pywt.idwt2((coeffs),'haar')
    return denoised

def segmentation(im):
    predicted = np.uint8(im>0.75*255)
    return predicted

def binary_open(im):
    kernel = np.ones((4,4),np.uint8)
    opened = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    return opened

def binary_close(im):
    kernel = np.ones((4,4),np.uint8)
    closed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    return closed

# https://github.com/vit1-irk/clean_lib
def makeClean(dirtysource, psf, cleanblurradius, bottomlimit, maxit, gamma = 0.1, criticalbottom = None):
    psfmax = psf.max()
    
    psfk, psfm = [ int((x-1) / 2) for x in psf.shape]
    imgx, imgy = dirtysource.shape
    imgbx, imgby = (int(imgx + 2*psfk), int(imgy + 2*psfm))

    img2big = Image.new("F", (imgbx, imgby), 0)
    img2big.paste(Image.fromarray(dirtysource), (psfk, psfm))

    sourcebytes = np.array(img2big, dtype=np.float64)
    totalmax = sourcebytes.max()

    if criticalbottom is None:
        criticalbottom = bottomlimit * totalmax

    dirtysum = np.sum(sourcebytes)

    cleanImage = Image.new("F", (imgbx, imgby), 0)
    cleanPoints = np.array(cleanImage, dtype=np.float64)

    psfsum = np.sum(psf)
    
    it = 0
    while it < maxit:
        maxpoint = np.unravel_index(sourcebytes.argmax(), sourcebytes.shape)
        y, x = maxpoint[:2]
        mpy, mpx = maxpoint[:2]

        maxvalue = sourcebytes[y][x]
        if maxvalue <= criticalbottom:
            break

        k = gamma * maxvalue / psfmax

        py = 0
        for y in range(mpy - psfm, mpy + psfm + 1):
            px = 0
            for x in range(mpx - psfk, mpx + psfk + 1):
                if y >= imgby or x >= imgbx or y < 0 or x < 0:
                    px += 1
                    continue
                sourcebytes[y][x] -= psf[py][px] * k
                px += 1
            py += 1

        cleanPoints[mpy][mpx] += psfsum * k

        if (it % 10) == 0:
            cleansum = np.sum(cleanPoints)
            dirtysum2 = np.sum(sourcebytes)
        it += 1
    print("iterations: {0}".format(it))
    
    cleanPoints = gaussian_filter(cleanPoints, sigma=cleanblurradius)
    cleanImage = Image.fromarray(cleanPoints)
    dirtyOutput = Image.fromarray(sourcebytes)

    cleanImage = cleanImage.crop((psfk, psfm, imgx + psfk, imgy + psfm))
    dirtyOutput = dirtyOutput.crop((psfk, psfm, imgx + psfk, imgy + psfm))

    return np.array(cleanImage), np.array(dirtyOutput)

# https://github.com/vit1-irk/clean_lib
def makeCleanBigPSF(dirtysource, psf, cleanblurradius, bottomlimit, maxit, gamma = 0.1, criticalbottom = None):
    psfmy, psfmx = np.unravel_index(psf.argmax(), psf.shape)[:2]

    imgy, imgx = dirtysource.shape
    totalmax = dirtysource.max()

    if criticalbottom is None:
        criticalbottom = bottomlimit * totalmax

    sourcebytes = dirtysource.copy()
    dirtysum = np.sum(sourcebytes)

    cleanImage = Image.new("F", (imgy, imgx), 0)
    cleanPoints = np.array(cleanImage, dtype=np.float64)

    it = 0
    while it < maxit:
        maxpoint = np.unravel_index(sourcebytes.argmax(), sourcebytes.shape)
        y, x = maxpoint[:2]

        maxvalue = sourcebytes[y][x]
        if maxvalue <= criticalbottom:
            break
        print(maxvalue)

        x0, x1 = (psfmx - x, psfmx - x + imgx)
        y0, y1 = (psfmy - y, psfmy - y + imgy)
        
        psf_slice = psf[y0:y1, x0:x1]

        psfsum = psf_slice.sum()
        psfmax = psf_slice.max()

        k = gamma * maxvalue / psfmax

        sourcebytes = sourcebytes - psf_slice * k
        cleanPoints[y][x] = cleanPoints[y][x] + psfsum * k

        if (it % 10) == 0:
            cleansum = np.sum(cleanPoints)
            dirtysum2 = np.sum(sourcebytes)
            print("full = {0}, sourcepic = {1}, it = {2}"
                  .format(cleansum + dirtysum2, dirtysum, it))

        it += 1
    print("iterations: {0}".format(it))
    
    cleanPoints = gaussian_filter(cleanPoints, sigma=cleanblurradius)

    return cleanPoints, sourcebytes

def create_ASC_PSF(img, fc, B, az):
    """
    Creates the point spread function to be used in the attributed scattering center (ASC) extraction process.
    
    The common ASC formula was being used to write the PSF creation code.
    
    Center frequency `fc` and bandwith `B` values are the same among all MSTAR data. Only the azimuth angle `az` changes.
    """
    # CLEAN parameters
    c = 3*1e8 #m/s, speed of light
    omega = az/180*np.pi # degrees->radians, azimuth aperture
    
    # Create point spread function
    h,w = img.shape
    img = resize(img, (h, h), preserve_range=True) # Some images have few pixels of offsets from being a square, s quick fix
    h,w = img.shape
    img.astype(np.float64)
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    x, y = np.meshgrid(x, y)

    # Use -35dB Taylor window since it was also used while the MSTAR dataset was being collected
    taylor_window = signal.windows.taylor(h, sll=35)
    t = taylor_window[x]*np.transpose(taylor_window[y])
    psf = np.exp(1j*4*np.pi*fc/c*((x-w//2)+15*np.pi/180*(y-w//2))) * (4*fc*B*omega/(c**2)) * np.sinc((2*B/c*(x-w//2))/np.pi) * np.sinc((2*fc*omega/c*(y-w//2))/np.pi)
    psf = psf * t
    psf_real = np.real(psf)
    
    return psf_real

def run_CLEAN(img, PSF, blurrad_w1=2, bottomlimit=0.01, gamma_w1=0.6):
    """
    Uses the CLEAN deconvolution algorithm from https://github.com/vit1-irk/clean_lib
    
    Cited study is focused on radioastronomy imaging, hence the used PSFs in that study are not suitable for SAR imaging
    
    A suitable PSF choice for SAR imaging would be attributed scattering center PSFs.
    """
    clean, dirtyoutput = makeClean(img, PSF, blurrad_w1, bottomlimit, maxit=100, gamma=gamma_w1, criticalbottom=None)
    return clean

def augment(img_path, ds_size=16):
    """
    Used by `create_dataset` function in data.py.
    `ds_size` is the width and height for downscaling input image.
    """
    img = io.imread(img_path)
    img = dwt_denoising(img)
    img = histogram_equalization(img)
    if ds_size is not None:
        img = resize(img, (ds_size, ds_size))
    features = img.flatten()
    return features

if __name__ == '__main__':
    img_path = "dataset/TEST_15/2S1/HB14931.ornt.JPG"
    features = augment(img_path)