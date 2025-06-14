import numpy as np
import pyaudio
import time
from scipy.signal import butter, lfilter
from collections import deque

# ความถี่เป้าหมายของสายกีตาร์
target_freqs = {
    "E2": 82.41,
    "A2": 110.00,
    "D3": 146.83,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 329.63
}

# ตั้งค่าเสียง
RATE = 44100
CHUNK = 2048

# Bandpass filter
def butter_bandpass(lowcut=80.0, highcut=350.0, fs=RATE, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, fs=RATE):
    b, a = butter_bandpass()
    return lfilter(b, a, data)

# Auto-correlation frequency detection
def detect_frequency_autocorr(signal, rate):
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    start = np.argmax(d > 0)
    if start == 0:
        return 0
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return 0
    return rate / peak

# หาชื่อโน้ตที่ใกล้ที่สุด
def freq_to_note_name(freq):
    closest_note = min(target_freqs, key=lambda note: abs(target_freqs[note] - freq))
    distance = freq - target_freqs[closest_note]
    return closest_note, target_freqs[closest_note], distance

# โปรแกรมหลัก
def listen_and_detect():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎸 เริ่มฟังเสียง... เล่นสายกีตาร์ทีละสาย")

    history = deque(maxlen=5)
    last_note = None
    last_time = time.time()

    while True:
        try:
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            filtered = bandpass_filter(data)

            rms = np.sqrt(np.mean(filtered**2))
            if rms < 500:
                continue

            freq = detect_frequency_autocorr(filtered, RATE)
            if not 80 < freq < 350:
                continue

            history.append(freq)
            if len(history) < history.maxlen:
                continue

            avg_freq = np.mean(history)
            note, target, diff = freq_to_note_name(avg_freq)

            # แสดงเฉพาะถ้าโน้ตเปลี่ยน หรือผ่านไป > 1 วิ
            if note != last_note or time.time() - last_time > 1:
                direction = "↑ หมุนให้สูงขึ้น" if diff < -1 else "↓ หมุนให้ต่ำลง" if diff > 1 else "✓ จูนแล้ว"
                print(f"🎵 {avg_freq:.2f} Hz → {note} ({target:.2f} Hz) {direction}")
                last_note = note
                last_time = time.time()
                time.sleep(3)

        except KeyboardInterrupt:
            print("🛑 หยุดการฟังแล้ว")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

listen_and_detect()
