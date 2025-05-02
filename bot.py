import cv2
import time
from rembg import remove
import wave
import pyttsx3
import speech_recognition as sr
import google.generativeai as genai
import datetime
import os
import numpy as np
from PIL import Image, ImageTk, ImageSequence
import glob
import threading
import subprocess
import tkinter as tk
from tkinter import ttk
from ffpyplayer.player import MediaPlayer
from gtts import gTTS
from pydub import AudioSegment


# 🔐 Configure Gemini API
genai.configure(api_key="AIzaSyCN62fHENF-KP6pTJSVtjr0GYUj8Qi48bg")
model = genai.GenerativeModel("gemini-2.0-flash")

PHONE_IP = "192.168.67.52"
PORT = "4747"
VIDEO_URL = f"http://{PHONE_IP}:{PORT}/video"

# 🎙️ TTS Initialization (Male Voice)
engine = pyttsx3.init()
engine.setProperty('rate', 160)
voices = engine.getProperty('voices')
for voice in voices:
    if "male" in voice.name.lower() or "david" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# GUI related globals
gif_running = False
gif_frames = []
gif_idx = 0
video_playing = False
root = None
canvas = None
main_frame = None
is_fullscreen = True


def get_audio_duration(filepath):
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception as e:
        print(f"❌ Error reading audio file duration: {e}")
        return 0
    
def speak(text):
    engine.say(text)
    engine.runAndWait()

def speak_and_save(text, filename="response.wav"):
    try:
        # Save as MP3 using gTTS
        tts = gTTS(text)
        mp3_path = "temp.mp3"
        tts.save(mp3_path)

        # Convert MP3 to WAV (if needed)
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(filename, format="wav")

        duration = get_audio_duration(filename)
        if duration > 1.5:
            print(f"✅ Audio generated: {duration:.2f} seconds")
            return generate_hologram_video("passport_no_bg.png", filename)
        else:
            print("❌ Audio too short.")
            return None
    except Exception as e:
        print(f"❌ Failed to generate speech: {e}")
        return None


def get_fresh_frame(cap):
    for _ in range(10):
        cap.grab()
    ret, frame = cap.read()
    return ret, frame

def capture_passport_image():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        user_input = input("Enter command (c=capture, q=quit): ").lower()
        if user_input == 'q':
            print("❌ Capture cancelled.")
            return False
        if user_input == 'c':
            print("📸 Activating camera...")
            cap = cv2.VideoCapture(VIDEO_URL)
            if not cap.isOpened():
                print("❌ Camera not reachable. Please check DroidCam.")
                return False
            ret, frame = get_fresh_frame(cap)
            cap.release()
            if not ret:
                print("❌ Failed to capture frame. Try again.")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                print("❌ No face detected. Try again.")
                continue
            print("✅ Face detected!")
            for (x, y, w, h) in faces:
                margin_x = int(w * 1.2)
                margin_y = int(h * 1.5)
                x_start = max(0, x - margin_x)
                y_start = max(0, y - margin_y)
                x_end = min(frame.shape[1], x + w + margin_x)
                y_end = min(frame.shape[0], y + h + int(h * 1.7))
                passport_photo = frame[y_start:y_end, x_start:x_end]
                resized = cv2.resize(passport_photo, (600, 450), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("passport_temp.jpg", resized)
                with open("passport_temp.jpg", "rb") as img_file:
                    input_image = img_file.read()
                output_image = remove(input_image)
                with open("passport_no_bg.png", "wb") as out_file:
                    out_file.write(output_image)
                print("✅ Image captured and background removed.")
            retry = input("Press 'r' to retry or press Enter to proceed: ").lower()
            if retry != 'r':
                return True

# def generate_hologram_video(image_path, audio_path="response.wav"):
#     screen_size = 1920
#     video_size = 400
#     input_path = 'template.mp4'
#     output_path = 'output.mp4'
#     final_output = 'output_with_audio.mp4'

#     cap = cv2.VideoCapture(input_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (screen_size, screen_size))

#     positions = {
#         "top": ((screen_size - video_size) // 2, 0),
#         "right": (screen_size - video_size, (screen_size - video_size) // 2),
#         "bottom": ((screen_size - video_size) // 2, screen_size - video_size),
#         "left": (0, (screen_size - video_size) // 2),
#     }
#     rotations = {"top": 0, "right": 270, "bottom": 180, "left": 90}

#     def rotate(img, angle):
#         center = (img.shape[1] // 2, img.shape[0] // 2)
#         matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#         return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

#     cap = cv2.VideoCapture(input_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         resized = cv2.resize(frame, (video_size, video_size))
#         canvas = np.zeros((screen_size, screen_size, 3), dtype=np.uint8)
#         for pos, (x, y) in positions.items():
#             rotated = rotate(resized, rotations[pos])
#             canvas[y:y+video_size, x:x+video_size] = rotated
#         out.write(canvas)

#     cap.release()
#     out.release()

#     subprocess.run([
#         "ffmpeg", "-y",
#         "-i", output_path,
#         "-i", audio_path,
#         "-map", "0:v:0",
#         "-map", "1:a:0",
#         "-c:v", "copy",
#         "-c:a", "aac",
#         "-b:a", "192k",
#         "-shortest",
#         final_output
#     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
#     return final_output

def generate_hologram_video(image_path, audio_path="response.wav"):
    """
    Generates a hologram-style video using deepfake inference followed by hologram effect
    
    Args:
        image_path: Path to the source image (passport photo)
        audio_path: Path to the audio file containing the AI response
    
    Returns:
        Path to the final output video with audio
    """
    screen_size = 1920
    video_size = 400
    results_dir = "./results"
    temp_template = f"{results_dir}/temp_template.mp4"
    output_path = 'output.mp4'
    final_output = 'output_with_audio.mp4'
    
    # Make sure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Run the deepfake inference script to generate the animated face video
    print("🧠 Running deepfake inference...")
    
    # Make the script executable
    os.chmod("run_inference.sh", 0o755)
    
    # Run inference script with our parameters
    subprocess.run([
        "./run_inference.sh",
        audio_path,  # Driven audio
        image_path,  # Source image
        results_dir  # Results directory
    ])
    
    # Find the most recent video file in the results directory (the generated deepfake)
    video_files = glob.glob(f"{results_dir}/*.mp4")
    if not video_files:
        print("❌ No video generated by deepfake inference. Using fallback.")
        input_path = 'template.mp4'  # Fallback to original template
    else:
        # Get most recently created video file
        input_path = max(video_files, key=os.path.getctime)
        print(f"✅ Using generated deepfake video: {input_path}")
    
    # Step 2: Create the hologram effect
    print("🎬 Creating hologram effect...")
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (screen_size, screen_size))

    positions = {
        "top": ((screen_size - video_size) // 2, 0),
        "right": (screen_size - video_size, (screen_size - video_size) // 2),
        "bottom": ((screen_size - video_size) // 2, screen_size - video_size),
        "left": (0, (screen_size - video_size) // 2),
    }
    rotations = {"top": 0, "right": 270, "bottom": 180, "left": 90}

    def rotate(img, angle):
        center = (img.shape[1] // 2, img.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    cap = cv2.VideoCapture(input_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (video_size, video_size))
        canvas = np.zeros((screen_size, screen_size, 3), dtype=np.uint8)
        for pos, (x, y) in positions.items():
            rotated = rotate(resized, rotations[pos])
            canvas[y:y+video_size, x:x+video_size] = rotated
        out.write(canvas)

    cap.release()
    out.release()

    # Step 3: Combine with audio
    print("🔊 Adding audio to hologram video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        final_output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("✅ Hologram video created successfully")
    return final_output

def load_gif(gif_path="assistant.gif"):
    """Load all frames of GIF into memory"""
    global gif_frames
    gif = Image.open(gif_path)
    gif_frames = [ImageTk.PhotoImage(img.copy().convert("RGBA").resize((canvas.winfo_width(), canvas.winfo_height()))) 
                 for img in ImageSequence.Iterator(gif)]
    return len(gif_frames)

def update_gif_frame():
    """Updates the GIF frame on canvas"""
    global gif_idx, gif_frames, canvas, gif_running
    
    if not gif_running or len(gif_frames) == 0:
        return
    
    # Clear canvas and draw new frame
    canvas.delete("all")
    canvas.create_image(canvas.winfo_width()//2, canvas.winfo_height()//2, 
                        image=gif_frames[gif_idx], anchor="center")
    
    # Update index for next frame
    gif_idx = (gif_idx + 1) % len(gif_frames)
    
    # Schedule next frame update
    if gif_running:
        root.after(100, update_gif_frame)

def start_gif_animation():
    """Start GIF animation on canvas"""
    global gif_running, canvas
    
    if not canvas:
        return
        
    # Make sure canvas has a size before loading GIF
    root.update_idletasks()
    
    gif_running = True
    load_gif()
    update_gif_frame()

def stop_gif_animation():
    """Stop GIF animation"""
    global gif_running
    gif_running = False

def setup_gui():
    """Setup the main GUI window with title bar and controls"""
    global root, canvas, main_frame, is_fullscreen
    
    # Create main window
    root = tk.Tk()
    root.title("AI Assistant")
    
    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Create a custom title bar frame
    title_bar = tk.Frame(root, bg='#2e2e2e', height=30)
    title_bar.pack(fill=tk.X)
    
    # Add title text
    title_label = tk.Label(title_bar, text="AI Assistant", bg='#2e2e2e', fg='white')
    title_label.pack(side=tk.LEFT, padx=10)
    
    # Add window control buttons
    minimize_button = tk.Button(title_bar, text="—", bg='#2e2e2e', fg='white', bd=0,
                               command=minimize_window)
    minimize_button.pack(side=tk.RIGHT)
    
    maximize_button = tk.Button(title_bar, text="⬜", bg='#2e2e2e', fg='white', bd=0,
                              command=toggle_fullscreen)
    maximize_button.pack(side=tk.RIGHT)
    
    close_button = tk.Button(title_bar, text="✕", bg='#2e2e2e', fg='white', bd=0,
                            command=root.destroy)
    close_button.pack(side=tk.RIGHT)
    
    # Main content frame
    main_frame = tk.Frame(root, bg='black')
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create canvas for displaying gif and video
    canvas = tk.Canvas(main_frame, bg='black', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Make window draggable from title bar
    title_bar.bind("<ButtonPress-1>", drag_start)
    title_bar.bind("<ButtonRelease-1>", drag_stop)
    title_bar.bind("<B1-Motion>", drag_motion)
    
    # Set window to fullscreen by default
    root.geometry(f"{screen_width}x{screen_height}")
    root.attributes('-fullscreen', is_fullscreen)
    
    return root

def drag_start(event):
    widget = event.widget
    widget._drag_start_x = event.x
    widget._drag_start_y = event.y

def drag_stop(event):
    widget = event.widget
    widget._drag_start_x = None
    widget._drag_start_y = None

def drag_motion(event):
    global is_fullscreen
    widget = event.widget
    if is_fullscreen:
        # If in fullscreen, switch to windowed mode first
        toggle_fullscreen()
    if (hasattr(widget, '_drag_start_x') and widget._drag_start_x is not None and
        hasattr(widget, '_drag_start_y') and widget._drag_start_y is not None):
        x = root.winfo_x() - widget._drag_start_x + event.x
        y = root.winfo_y() - widget._drag_start_y + event.y
        root.geometry(f"+{x}+{y}")

def toggle_fullscreen():
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    root.attributes('-fullscreen', is_fullscreen)

def minimize_window():
    root.iconify()

def play_video(video_path="output_with_audio.mp4"):
    """Play video in the same canvas"""
    global canvas, gif_running, video_playing
    
    # Stop GIF animation
    stop_gif_animation()
    
    # Clear canvas
    canvas.delete("all")
    
    video_playing = True
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def update_video():
        nonlocal cap, player
        global video_playing
        
        if not video_playing:
            # Clean up
            cap.release()
            player.close_player()
            # Restart GIF when video ends
            start_gif_animation()
            return
            
        ret, frame = cap.read()
        audio_frame, val = player.get_frame()
        
        if not ret:
            # Video finished
            video_playing = False
            cap.release()
            player.close_player()
            # Restart GIF when video ends
            start_gif_animation()
            return
            
        # Convert frame for display
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Resize to fill canvas while maintaining aspect ratio
        frame = cv2.resize(frame, (canvas_width, canvas_height))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Display frame
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk, anchor="center")
        canvas.img_tk = img_tk  # Keep reference to prevent garbage collection
        
        # Schedule next frame update
        root.after(30, update_video)
    
    # Start video playback
    update_video()

def gemini_voice_assistant():
    """Run the voice assistant logic"""
    global video_playing
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🎤 Ask a question (or say 'exit', 'stop', 'quit', 'bye' to end)...")
    
    while True:
        # Skip listening if video is playing
        if video_playing:
            time.sleep(0.5)  # Short sleep to reduce CPU usage
            continue
            
        print("🎤 Listening...")
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5)  # Add timeout to prevent hanging
                
            query = recognizer.recognize_google(audio).lower()
            print(f"🤔 Question: {query}")
            
            if query in ["exit", "stop", "quit", "bye"]:
                print("👋 Exiting voice assistant.")
                break
                
            # Add prompt to limit response to 50 words
            prompt = f"Answer this question within 50 words: {query}"
            response = model.generate_content(prompt)
            reply_text = response.text.strip()
            print(f"🤖 Gemini says: {reply_text}")
            
            output_path = speak_and_save(reply_text)
            if output_path:
                play_video(output_path)
            
        except sr.WaitTimeoutError:
            # Timeout occurred, continue listening
            pass
        except sr.UnknownValueError:
            # Speech wasn't understood
            print("❌ Sorry, I didn't catch that.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("🔄 Please try again.")

def start_assistant():
    """Start the assistant GUI and logic"""
    global root
    
    # Setup GUI
    setup_gui()
    
    # Start GIF animation after a short delay
    root.after(500, start_gif_animation)
    
    # Start voice assistant in a separate thread
    assistant_thread = threading.Thread(target=gemini_voice_assistant)
    assistant_thread.daemon = True
    assistant_thread.start()
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    try:
        print("🚀 Starting passport photo capture...")
        if capture_passport_image():
            print("✅ Photo captured successfully. Starting voice assistant GUI...")
            start_assistant()
        else:
            print("❌ Photo capture failed or cancelled.")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting.")
        stop_gif_animation()
        if root:
            root.destroy()