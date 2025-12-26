import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter

class HPSSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HPSS - Median Filter Separator")
        self.root.geometry("500x350")
        self.root.resizable(False, False)

        # 상태 변수
        self.file_path = None
        self.is_processing = False

        # UI 구성
        self._setup_ui()

    def _setup_ui(self):
        # 제목
        title_label = tk.Label(self.root, text="Harmonic/Percussive Separator", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=20)

        # 파일 선택 영역
        self.file_label = tk.Label(self.root, text="선택된 파일 없음", fg="gray", wraplength=400)
        self.file_label.pack(pady=10)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.select_btn = tk.Button(btn_frame, text="📂 오디오 파일 선택", command=self.select_file, width=20, height=2)
        self.select_btn.pack()

        # 실행 버튼
        self.process_btn = tk.Button(self.root, text="🚀 분리 시작 (Start HPSS)", command=self.start_processing, 
                                     width=20, height=2, bg="#007bff", fg="white", state="disabled")
        self.process_btn.pack(pady=20)

        # 상태 메시지
        self.status_label = tk.Label(self.root, text="파일을 선택해주세요.", fg="blue")
        self.status_label.pack(pady=10)

    def select_file(self):
        filetypes = (("Audio files", "*.mp3 *.wav *.flac *.m4a"), ("All files", "*.*"))
        filename = filedialog.askopenfilename(title="오디오 파일 열기", initialdir="/", filetypes=filetypes)
        
        if filename:
            self.file_path = filename
            self.file_label.config(text=os.path.basename(filename), fg="black")
            self.process_btn.config(state="normal", bg="#007bff")
            self.status_label.config(text="준비 완료. '분리 시작'을 누르세요.")

    def start_processing(self):
        if not self.file_path:
            return
        
        # GUI 프리징 방지를 위해 쓰레딩 사용
        self.is_processing = True
        self.process_btn.config(state="disabled", text="처리 중... (잠시만 기다려주세요)")
        self.select_btn.config(state="disabled")
        
        thread = threading.Thread(target=self.run_hpss_algorithm)
        thread.start()

    def run_hpss_algorithm(self):
        try:
            # === [논문 구현 핵심부] ===
            
            # 1. 오디오 로드 및 STFT 변환
            # 논문 추천: FFT size 4096 (저음 해상도 확보) 
            y, sr = librosa.load(self.file_path, sr=None)
            S_full = librosa.stft(y, n_fft=4096, hop_length=1024)
            
            # 크기(Magnitude)와 위상(Phase) 분리
            S_mag, S_phase = librosa.magphase(S_full)

            # 2. 미디언 필터 적용 (Median Filtering)
            # 논문: Harmonic은 가로(Horizontal), Percussive는 세로(Vertical) 특성을 가짐 [cite: 7, 20]
            # 논문 추천 커널 크기: 15 ~ 30 사이 (여기선 31 사용) [cite: 173]
            kernel_size = 31
            
            # 가로 필터 (Harmonic 강화): (1, kernel_size) -> 시간 축으로 스무딩
            H_filter = median_filter(S_mag, size=(1, kernel_size))
            
            # 세로 필터 (Percussive 강화): (kernel_size, 1) -> 주파수 축으로 스무딩
            P_filter = median_filter(S_mag, size=(kernel_size, 1))

            # 3. 소프트 마스킹 (Soft Masking via Wiener Filtering)
            # 논문 수식 (11): M_H = H^p / (H^p + P^p) (p=2 추천) [cite: 160, 176]
            p = 2
            H_pow = H_filter ** p
            P_pow = P_filter ** p
            total_pow = H_pow + P_pow + 1e-10 # 0으로 나누기 방지

            M_H = H_pow / total_pow
            M_P = P_pow / total_pow

            # 4. 원본 스펙트로그램에 마스크 적용
            H_sep = S_mag * M_H
            P_sep = S_mag * M_P

            # 5. iSTFT (다시 오디오로 변환) - 위상(Phase) 정보 복원
            y_harmonic = librosa.istft(H_sep * S_phase, hop_length=1024)
            y_percussive = librosa.istft(P_sep * S_phase, hop_length=1024)

            # === [결과 저장] ===
            base_name = os.path.splitext(self.file_path)[0]
            sf.write(f"{base_name}_harmonic.wav", y_harmonic, sr)
            sf.write(f"{base_name}_percussive.wav", y_percussive, sr)

            self.root.after(0, lambda: self.finish_processing(True, base_name))

        except Exception as e:
            self.root.after(0, lambda: self.finish_processing(False, str(e)))

    def finish_processing(self, success, message):
        self.is_processing = False
        self.process_btn.config(state="normal", text="🚀 분리 시작 (Start HPSS)")
        self.select_btn.config(state="normal")

        if success:
            self.status_label.config(text="완료! 원본 파일 위치에 저장되었습니다.")
            messagebox.showinfo("성공", f"분리가 완료되었습니다!\n\n저장 위치:\n{message}_harmonic.wav\n{message}_percussive.wav")
        else:
            self.status_label.config(text="오류 발생")
            messagebox.showerror("에러", f"처리 중 문제가 발생했습니다:\n{message}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HPSSApp(root)
    root.mainloop()