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

import numpy as np
 
SR = 16000            # resample everything; librosa's sr=None leaves files at
                      # their native rate, and then a hop in samples is a
                      # different amount of time per file
WIN_S = 0.025         # 25 ms analysis window, standard for speech
HOP_S = 0.010         # 10 ms hop -> 100 Hz frame rate
CONTEXT_S = 0.030     # extra audio either side so edge windows are full
N_MELS = 40
FMIN, FMAX = 50, 8000
 
 
# ==============================================================================
# Loading
# ==============================================================================
 
def load_segment(path, start, end, sr=SR, context=CONTEXT_S):
    """
    Returns (y, sr, t0) where t0 is the absolute time of y[0]. Loads a little
    beyond the segment so that windows near the edges are complete.
    """
    import librosa
    offset = max(0.0, start - context)
    duration = (end + context) - offset
    if path.endswith(".npy"):
        raw = np.load(path).flatten()
        file_sr = int(np.asarray(np.load(path.replace("_audio.npy", "_sr.npy"))).ravel()[0])
        # only integer PCM needs the /32768; dividing a float array destroys it,
        # and a mixed corpus would then have two different amplitude scales,
        # which shows up directly in c0
        if np.issubdtype(raw.dtype, np.integer):
            raw = raw.astype(np.float32) / 32768.0
        else:
            raw = raw.astype(np.float32)
        a = max(0, int(round(offset * file_sr)))
        b = min(len(raw), int(round((offset + duration) * file_sr)))
        y = raw[a:b]
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    else:
        y, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
    return np.asarray(y, dtype=np.float32), sr, offset
 
 
# ==============================================================================
# MFCCs
# ==============================================================================
 
def get_mfccs(path, start, end, n_mfcc=13, drop_c0=True, sr=SR,
              win_s=WIN_S, hop_s=HOP_S, min_frames=3, raise_errors=False):
    """
    Returns (T, F) float32 with T proportional to duration, or None.
 
    n_mfcc=13 is the speech-processing default. 30 is not wrong, but coefficients
    beyond ~20 of 40 mel bands are mostly numerical noise, and each one is
    another dimension the MSE has to spend on.
 
    drop_c0=True discards the zeroth cepstral coefficient, which is essentially
    log energy and therefore tracks recording level, mic distance and how loudly
    a speaker happened to talk. Keep it only if loudness is part of what you
    mean by the realisation.
 
    `raise_errors=True` while you are setting up: a bare `except: return None`
    hides why tokens are being dropped, and dropped tokens matter later when you
    intersect the acoustic and articulatory sets for RSA.
    """
    import librosa
    try:
        y, sr, t0 = load_segment(path, start, end, sr=sr)
        n_fft = int(round(win_s * sr))
        hop = int(round(hop_s * sr))
        if len(y) < n_fft:
            return None
        n = n_mfcc + 1 if drop_c0 else n_mfcc
        M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n, n_fft=n_fft, hop_length=hop,
                                 n_mels=N_MELS, fmin=FMIN, fmax=min(FMAX, sr // 2),
                                 center=True)
        if drop_c0:
            M = M[1:]
        # keep frames whose CENTRE falls inside the segment
        centres = t0 + librosa.frames_to_time(np.arange(M.shape[1]), sr=sr, hop_length=hop)
        M = M[:, (centres >= start) & (centres < end)]
        if M.shape[1] < min_frames:
            return None
        return np.ascontiguousarray(M.T, dtype=np.float32)      # (T, F)
    except Exception:
        if raise_errors:
            raise
        return None