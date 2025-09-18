import numpy as np
import nibabel as nib
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import exposure

def PSNR_and_SSIM(path1, path2, mu=1, sigma=0.1, plot=True):
    """
    Computes PSNR and SSIM between two images in .nii or .dcm format after adding Rician noise to the first image.
    Args:
        path1 (str): Path to the first image (.nii or .dcm)
        path2 (str): Path to the second image (.nii or .dcm)
        mu (float): Mean of the Rician noise to be added
        sigma (float): Standard deviation of the Rician noise to be added
        plot (bool): If True, plots the two images side by side with metrics annotated below. If False, only returns the metrics.
    Returns:
        tuple: (MSE, PSNR, SSIM) between the two images.
    """
    
    # Load images
    if path1.endswith('.dcm'):
        img1 = pydicom.dcmread(path1).pixel_array
    else:
        img1 = nib.load(path1).get_fdata()

    if path2.endswith('.dcm'):
        img2 = pydicom.dcmread(path2).pixel_array
    else:
        img2 = nib.load(path2).get_fdata()

    # Normalize images for comparison
    img1_norm = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    img2_norm = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

    # flip img1 horizontally and rotate 90 degrees counter-clockwise
    if path1.endswith('.nii'):
        img1_norm = np.fliplr(np.rot90(img1_norm, k=-1))
    else:
        pass

    # Apply Fourier transform to img1_norm
    img1_fft = np.fft.fft2(img1_norm)
    img1_fft_shifted = np.fft.fftshift(img1_fft)

    # Add Rician noise in the frequency domain
    real_noise = np.random.normal(mu, sigma, img1_fft_shifted.shape)
    imag_noise = np.random.normal(mu, sigma, img1_fft_shifted.shape)
    noisy_fft = img1_fft_shifted + (real_noise + 1j * imag_noise)

    # Inverse Fourier transform to get back to image domain
    img1_noisy = np.abs(np.fft.ifft2(np.fft.ifftshift(noisy_fft)))


    # Normalize the noisy image
    img1_norm = (img1_noisy - np.min(img1_noisy)) / (np.max(img1_noisy) - np.min(img1_noisy))

    # Compute metrics
    mse = mean_squared_error(img1_norm, img2_norm)
    psnr = peak_signal_noise_ratio(img1_norm, img2_norm, data_range=1.0)
    ssim = structural_similarity(img1_norm, img2_norm, data_range=1.0)

    print(f'MSE: {mse}, PSNR: {psnr}, SSIM: {ssim}')

    if plot == False:
        return mse, psnr, ssim
    else:
        pass
    # Plot images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1_norm, cmap='gray')
    axes[0].set_title('Synthetic LR', fontsize=18)
    axes[0].axis('off')
    axes[1].imshow(img2_norm, cmap='gray')
    axes[1].set_title('Dynamic speech', fontsize=18)
    axes[1].axis('off')

    # Annotate metrics below images
    plt.figtext(0.5, 0.01, f'MSE: {mse:.4f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}', ha='center', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

    return mse, psnr, ssim



# Paths to your .nii files
file1 = 'Data/Synthetic LR/LR_000.nii'
file2 = 'dynamic_acqstn_example.dcm'

# Call the function
PSNR_and_SSIM(file1, file2, mu=1, sigma=0.2, plot=True)

