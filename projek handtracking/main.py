import cv2
import mediapipe as mp

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka.")
    raise SystemExit

# Konfigurasi deteksi tangan
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            print("Gagal membaca frame kamera.")
            break

        # Membuat tampilan seperti cermin
        frame = cv2.flip(frame, 1)

        # OpenCV menggunakan BGR, MediaPipe menggunakan RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Deteksi tangan
        results = hands.process(rgb_frame)

        finger_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Gambar titik dan garis tangan
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = hand_landmarks.landmark

                # Indeks ujung dan sendi setiap jari
                finger_tips = [8, 12, 16, 20]
                finger_pips = [6, 10, 14, 18]

                # Hitung empat jari selain ibu jari
                for tip, pip in zip(
                    finger_tips,
                    finger_pips
                ):
                    if landmarks[tip].y < landmarks[pip].y:
                        finger_count += 1

                # Hitung ibu jari berdasarkan posisi horizontal.
                # Pendekatan ini sesuai untuk orientasi tangan tertentu.
                thumb_tip = landmarks[4]
                thumb_ip = landmarks[3]

                if thumb_tip.x < thumb_ip.x:
                    finger_count += 1

        # Tampilkan angka
        cv2.putText(
            frame,
            f"Finger Count: {finger_count}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        # Tampilkan video
        cv2.imshow("Hand Tracking Project", frame)

        # Tekan Q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()