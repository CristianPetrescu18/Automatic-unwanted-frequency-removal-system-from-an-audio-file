import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import scipy as sp
from scipy.signal import butter, lfilter

plt.close('all')

def plot_magnitude_spectrum(signal, title, sr, f_ratio=1): #analiza spectrala
    FFT = sp.fft.fft(signal)
    magnitude_spectrum = np.abs(FFT)
    freqs = np.fft.fftfreq(len(FFT), 1/sr)
    
    return (FFT,magnitude_spectrum, freqs)

def fundamental_frequency(fft_result, freqs): #Detectare Frecvențe Nedorite și Calcul Frecvență Fundamentală
# Select frequencies in the desired range (80Hz - 630Hz)
    mask = (freqs >= 80) & (freqs <= 400)
    selected_freqs = freqs[mask]
    selected_fft_result = fft_result[mask]

    freq0 = np.abs(selected_freqs[np.argmax(np.abs(selected_fft_result))])
    harmonics = [freq0 * i for i in range(2, 6)]  # de la armonica 2 la 6 

    return (freq0, harmonics)

def butter_bandpass(lowcut, highcut, fs, order=4):#Filtrare Band-pass pentru Armonice pe Semnalul Original
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):#Filtrare Band-pass pentru Partea Vocală pe Semnalul Original
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def process_audio_file(audio_file):
#------ Încărcarea fișierelor audio ------------------------
    audio_signal, sr = librosa.load(audio_file)
    
#------Spectru magnitudine folosind FFT------------------------
    fft_result, mag_1, freqs = plot_magnitude_spectrum(audio_signal, "Spectru", sr,0.5)

#---------Frecventa fundamentala aflata in banda vorbirii---------------------
    freq0, harmonics = fundamental_frequency(fft_result, freqs)
    
#--------- Definește band-pass filter parameters pentru armonice---------------
    harmonics_lowcut = min(harmonics) - 10  # Add some margin
    harmonics_highcut = max(harmonics) + 10  # Add some margin
    harmonics_order = 4  # Filter order

#--------- Definește band-pass filter parameters pentru partea vocală---------------
    vocal_lowcut = 80
    vocal_highcut = 400
    vocal_order = 4

# ----------Aplică band-pass filter pentru armonice----------------------------------
    harmonics_filtered_data = butter_bandpass_filter(audio_signal, harmonics_lowcut, harmonics_highcut, sr, harmonics_order)

#---------- Aplică band-pass filter pentru partea vocală-----------------------------
    vocal_filtered_data = butter_bandpass_filter(audio_signal, vocal_lowcut, vocal_highcut, sr, vocal_order)

#---------- Combina rezultatele filtrărilor pentru a obține semnalul final--------------
    final_signal = harmonics_filtered_data + vocal_filtered_data


# Salvează fișierul audio modificat
    file_name, file_extension = os.path.splitext(os.path.basename(audio_file))
    filtered_file_name = f"{file_name}_filtered{file_extension}"
    filtered_file_path = os.path.join(os.path.dirname(audio_file), filtered_file_name)

# Convert the signal to integer format before writing
    final_signal_int = (final_signal * 32767).astype(np.int16)

# Use scipy to write the WAV file
    sp.io.wavfile.write(filtered_file_path, sr, final_signal_int) 


def select_file():
    global audio_file
    audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    file_label.config(text=f"Selected File: {os.path.basename(audio_file)}")
    return audio_file
    

def play_original():
    global audio_signal, sr
    audio_signal, sr = librosa.load(audio_file)
    sd.play(audio_signal, sr)
    sd.wait()
    
def download_file():
    global audio_file
    process_audio_file(audio_file)


def play_filtered():
    global filtered_file_path
    file_name, file_extension = os.path.splitext(os.path.basename(audio_file))
    filtered_file_name = f"{file_name}_filtered{file_extension}"
    filtered_file_path = os.path.join(os.path.dirname(audio_file), filtered_file_name)

    final_signal, sr = librosa.load(filtered_file_path)
    sd.play(final_signal, sr)
    sd.wait()

# UI
root = tk.Tk()
root.title("Microphone feedback")

select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack()

file_label = tk.Label(root, text="Selected File: None")
file_label.pack()

visualize_button = tk.Button(root, text="Download File", command=download_file)
visualize_button.pack()

play_original_button = tk.Button(root, text="Play Original", command=play_original)
play_original_button.pack()

play_filtered_button = tk.Button(root, text="Play Filtered", command=play_filtered)
play_filtered_button.pack()


root.mainloop()