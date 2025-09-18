import numpy as np

def add_rician_noise(img, mu=0, sigma=0.1):
    """
    Adds Rician noise to an image in the frequency domain.

    Args:
        img (numpy.ndarray): Input image.
        mu (float): Mean of the Rician noise.
        sigma (float): Standard deviation of the Rician noise.

    Returns:
        numpy.ndarray: Image with Rician noise added.
    """
    # Apply Fourier transform
    img_fft = np.fft.fft2(img)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # Add Rician noise in the frequency domain
    real_noise = np.random.normal(mu, sigma, img_fft_shifted.shape)
    imag_noise = np.random.normal(mu, sigma, img_fft_shifted.shape)
    noisy_fft = img_fft_shifted + (real_noise + 1j * imag_noise)

    # Inverse Fourier transform to get back to image domain
    img_noisy = np.abs(np.fft.ifft2(np.fft.ifftshift(noisy_fft)))

    return img_noisy