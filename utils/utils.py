# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:43:28 2018
@author: eesungkim
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def spectrogram(signal, NFFT, hop_length, window='hann'):
    return np.abs(librosa.stft(signal, n_fft=NFFT, hop_length=hop_length, window=window))**2

def LogPowerSpectrum(signal, NFFT, hop_length, window='hann'):
    return np.log(spectrogram(signal, NFFT, hop_length, window=window)) 

def show_signal(signal,signal2,signal_reconstructed,sr):
    plt.figure(figsize=(10,10))
    
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(signal, sr=sr)
    plt.title('Clean Time Signal')

    plt.subplot(3, 1, 2)
    librosa.display.waveplot(signal2, sr=sr)
    plt.title('Noisy Time Signal')

    plt.subplot(3, 1, 3)
    librosa.display.waveplot(signal_reconstructed, sr=sr)
    plt.title('Reconstructed Clean Time Signal')

def show_spectrogram(signal,signal2,recnstrtSignal,sr,NFFT,hop_length):
    #Display power (energy-squared) spectrogram
    plt.figure(figsize=(10,10))
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=NFFT, hop_length=hop_length),ref=np.max),sr=sr, x_axis='time', y_axis='linear')
    #librosa.display.specshow(librosa.amplitude_to_db(origianlSpectrogram, ref_power=np.max),sr=sr, x_axis='time',y_axis='linear')
    plt.title('Clean Spectrogram')
    plt.colorbar(format='%+02.0f dB')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=signal2, sr=sr, n_fft=NFFT, hop_length=hop_length),ref=np.max),sr=sr, x_axis='time', y_axis='linear')
    #librosa.display.specshow(librosa.amplitude_to_db(origianlSpectrogram, ref_power=np.max),sr=sr, x_axis='time',y_axis='linear')
    plt.title('Noisy Spectrogram')
    plt.colorbar(format='%+02.0f dB')
    
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=recnstrtSignal, sr=sr, n_fft=NFFT, hop_length=hop_length),ref=np.max),sr=sr, x_axis='time', y_axis='linear')
    #librosa.display.specshow(librosa.amplitude_to_db(recnstrtSpectrogram, ref_power=np.max),sr=sr, x_axis='time', y_axis='linear')
    plt.title('Reconstructed Clean Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
def divide_magphase(D, power=1):
    """Separate a complex-valued stft D into its magnitude (S)
    and phase (P) components, so that `D = S * P`."""
    mag = np.abs(D)
    mag **= power
    phase = np.exp(1.j * np.angle(D))

    return mag, phase

def merge_magphase(magnitude, phase):
    """merge magnitude (S) and phase (P) to a complex-valued stft D so that `D = S * P`."""
    # magnitude * np.exp(np.angle(D)*1j)
    # magnitude * numpy.cos(np.angle(D))+ magnitude * numpy.sin(np.angle(D))*1j
    return magnitude * phase
