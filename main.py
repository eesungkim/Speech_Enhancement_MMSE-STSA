"""
Created on Tue May  1 20:43:28 2018
@author: eesungkim
"""
import os
import math

import librosa
import argparse
import numpy as np
import scipy.io.wavfile as wav
from utils.estnoise_ms import * 
from utils.utils import * 

def MMSE_STSA(args):
    """Speech Enhancement using A Spectral Amplitude Estimator
    """
    PATH_ROOT = os.getcwd() 
    path_clean_test = os.path.join(PATH_ROOT , args.input_clean)
    path_noisy_test = os.path.join(PATH_ROOT , args.input_noisy)
    output_path_estimated_noisy_test = os.path.join(PATH_ROOT , args.output_file)
     
    (sr, clean_test) = wav.read(path_clean_test)
    (sr, noisy_test) = wav.read(path_noisy_test)

    maxPosteriorSNR= 100   
    minPosteriorSNR= 1
    
    #NFFT=256 
    #hop_length_sample = 128 
    #winfunc = 'hamming'     
    NFFT=args.num_FFT
    hop_length_sample = args.hop_size
    winfunc = args.window
    
    smoothFactorDD=0.99

    # the variance of the speech; lambda_x(k)
    #noisy
    stft_noisy_test = librosa.stft(noisy_test, n_fft=NFFT, hop_length=hop_length_sample, window=winfunc)   
    magnitude_noisy_test, phase_noisy_test = divide_magphase(stft_noisy_test, power=1)
        
    pSpectrum = magnitude_noisy_test**2                   
    
    # estimate the variance of the noise using minimum statistics noise PSD estimation ; lambda_d(k). 
    estNoise = estnoisem(pSpectrum,hop_length_sample/sr)     
    estNoise = estNoise
    
    aPosterioriSNR=pSpectrum/estNoise                    
    aPosterioriSNR=aPosterioriSNR
    aPosterioriSNR[aPosterioriSNR > maxPosteriorSNR] = maxPosteriorSNR
    aPosterioriSNR[aPosterioriSNR < minPosteriorSNR] = minPosteriorSNR

    previousGainedaPosSNR=1 
    (nFrames,nFFT2) = pSpectrum.shape                
    totalGain =[]
    for i in range(nFFT2):                         
        aPosterioriSNR_frame = aPosterioriSNR[:,i]                  
        
        #operator [2](52)
        oper=aPosterioriSNR_frame-1
        oper[oper < 0] = 0 
        smoothed_a_priori_SNR = smoothFactorDD * previousGainedaPosSNR + (1-smoothFactorDD) * oper
        
        #V for MMSE estimate ([2](8)) 
        V=smoothed_a_priori_SNR*aPosterioriSNR_frame/(1+smoothed_a_priori_SNR)            
        
        #Calculate Gain function which results from the MMSE [2](7),(12).
        gain= smoothed_a_priori_SNR/(1+smoothed_a_priori_SNR)  
        if any(V<1):
            gain[V<1] = (math.gamma(1.5) * np.sqrt(V[V<1])) / aPosterioriSNR_frame[V<1] * np.exp(-1 * V[V<1] / 2) * ((1 + V[V<1]) * bessel(0, V[V<1] / 2) + V[V<1] * bessel(1, V[V<1] / 2))
        
        previousGainedaPosSNR = (gain**2) * aPosterioriSNR_frame
        totalGain.append(gain)
    
    totalGain=np.array(totalGain)

    magnitude_estimated_clean = totalGain.T * magnitude_noisy_test
    stft_reconstructed_clean = merge_magphase(magnitude_estimated_clean, phase_noisy_test)
    signal_reconstructed_clean =librosa.istft(stft_reconstructed_clean, hop_length=hop_length_sample, window=winfunc)
    signal_reconstructed_clean=signal_reconstructed_clean.astype('int16')
    
    wav.write(output_path_estimated_noisy_test,sr,signal_reconstructed_clean)
    
    #display signals, spectrograms
    show_signal(clean_test,noisy_test,signal_reconstructed_clean,sr)
    show_spectrogram(clean_test,noisy_test, signal_reconstructed_clean,sr,NFFT,hop_length_sample)
    
def parse_args():
    parser = argparse.ArgumentParser(description='MMSE-STSA Speech Enhancement')
    parser.add_argument('--datasets_dir', type=str, default='datasets/',
                        help='')
    parser.add_argument('--input_clean', type=str, default='datasets/clean.wav',
                        help='datasets/clean_file_name.wav')
    parser.add_argument('--input_noisy', type=str, default='datasets/noisy_white_3dB.wav',
                        help='datasets/noisy_file_name.wav')
    parser.add_argument('--output_file', type=str, default='datasets/clean_estimated_MMSE_STSA_test.wav',
                        help='datasets/output_file_name.wav')
    parser.add_argument('--num_FFT', type=int, default='256',
                        help='')
    parser.add_argument('--hop_size', type=int, default='128',
                        help='')
    parser.add_argument('--window', type=str, default='hamming',
                        help='')
    return check_args(parser.parse_args())

def check_args(args):
    if not os.path.exists(args.datasets_dir):
        os.makedirs(args.datasets_dir)
    assert args.num_FFT >= 1, 'number of FFT size must be larger than or equal to one'
    assert args.hop_size < args.num_FFT, 'hop size must be smaller than number of FFT size'
    return args

if __name__ == '__main__':
    args = parse_args()
    MMSE_STSA(args)

    
          


