import librosa

def get_mfccs(wav_file_path, start_time, end_time, n_mfcc=30, n_frames=10):
    # === LOAD AUDIO ===
    # Load only the portion of interest
    duration = end_time - start_time
    if wav_file_path.endswith(".wav"):
        y, sr = librosa.load(wav_file_path, sr=None, offset=start_time, duration=duration)
    elif wav_file_path.endswith(".npy"):
        import numpy as np
        y = np.load(wav_file_path).flatten()
        sr = np.load(wav_file_path.replace("_audio.npy", "_sr.npy"))
        y = y[round(start_time*sr):round(end_time*sr)]
        y = y.astype(np.float32) / 32768.0

    hop_length = y.shape[0]//n_frames

    # === COMPUTE MFCCs ===
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    except Exception as e:
        return None
    mfccs = mfccs[:, :n_frames]

    return mfccs