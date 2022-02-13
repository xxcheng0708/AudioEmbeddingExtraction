#coding:utf-8
"""
    Created by cheng star at 2022/1/16 20:16
    @email : xxcheng0708@163.com
"""
import librosa
import os
import numpy as np
import resampy

DEFAULT_FS = 32000


def read_audio_file(filename):
    try:
        y, sr = librosa.load(filename, sr=None)
        if sr != DEFAULT_FS:
            y = resampy.resample(y, sr, DEFAULT_FS)
            sr = DEFAULT_FS
        return y, sr
    except Exception as e:
        print(e)


def split_audio(x, split_duration=5, step_duration=3, sample_count=20):
    try:
        std_duration = split_duration * DEFAULT_FS
        std_step = step_duration * DEFAULT_FS
        d_len = x.shape[1]

        if x.shape[1] < std_duration:
            out = np.repeat(x, std_duration // x.shape[1] + 1, axis=1)
            print("out shape : {}".format(out.shape))
            return [out[:, :std_duration]]
        else:
            result = []
            for s_len in range(0, d_len, std_step):
                print("s_len: {}".format(s_len))
                if d_len - s_len < std_duration:
                    new_data_array = x[:, -std_duration]
                else:
                    new_data_array = x[:, s_len: s_len + std_duration]
                result.append(new_data_array)
            return result[:sample_count]
    except Exception as e:
        print(e)


def mel_frequency(x, fs):
    print("x shape: {}".format(x.shape))

    mel_spectrogram_librosa = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=1024, hop_length=320, n_mels=64,
                                                             fmin=50, fmax=14000, power=1)
    print("shape: {}, min value: {}, max value: {}".format(mel_spectrogram_librosa.shape,
                                                           np.min(mel_spectrogram_librosa),
                                                           np.max(mel_spectrogram_librosa)))
    mel_spectrogram_librosa_db = librosa.power_to_db(mel_spectrogram_librosa, ref=1, top_db=120)
    print("db shape: {}, min value: {}, max value: {}".format(mel_spectrogram_librosa_db.shape,
                                                              np.min(mel_spectrogram_librosa_db),
                                                              np.max(mel_spectrogram_librosa_db)))
    return mel_spectrogram_librosa_db


def get_mel_spec(audio_path, split_duration, step_duration, sample_count, out_path):
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    dest_path = os.path.join(out_path, audio_basename)
    if os.path.exists(dest_path) is False:
        os.makedirs(dest_path)

    audio, fs = read_audio_file(audio_path)
    if audio is None:
        return

    audio = audio[np.newaxis, :]
    result = split_audio(audio, split_duration=split_duration, step_duration=step_duration, sample_count=sample_count)
    print("len result: {}".format(len(result)))

    for i, res in enumerate(result):
        x = np.squeeze(res)
        mel_spectrogram_librosa_db = mel_frequency(x, fs)
        print("mel_spectrogram_librosa_db: {}".format(mel_spectrogram_librosa_db.shape))

        np.save(os.path.join(dest_path, "{}_{}.npy".format(audio_basename, i)), mel_spectrogram_librosa_db)


if __name__ == '__main__':
    audio_path = "./data/a.mp4"
    get_mel_spec(audio_path, split_duration=5, step_duration=3, sample_count=20, out_path="./results/mel_features")