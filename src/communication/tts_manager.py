"""
Text-to-Speech Module
Handles TTS announcements and feedback
"""
import pyttsx3
import threading
import time

class TTSManager:
    def __init__(self):
        """Initialize Text-to-Speech manager"""
        self.tts_engine = self.init_tts()
        self.tts_lock = threading.Lock()
        
        # TTS cooldown settings
        self.last_eye_tts_time = 0
        self.last_phone_tts_time = 0
        self.last_gaze_tts_time = 0
        self.eye_tts_cooldown = 4
        self.phone_tts_cooldown = 5
        self.gaze_tts_cooldown = 3
    
    def init_tts(self):
        """Initialize Text-to-Speech engine"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 140)
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            print("✅ TTS engine initialized")
            return engine
        except Exception as e:
            print(f"❌ TTS initialization failed: {e}")
            return None
    
    def speak_async(self, text):
        """Speak text asynchronously"""
        def speak():
            with self.tts_lock:
                if self.tts_engine:
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                    except Exception as e:
                        print(f"TTS error: {e}")
        
        if self.tts_engine:
            threading.Thread(target=speak, daemon=True).start()
    
    def handle_eye_alert(self, eye_results):
        """Handle eye-related TTS alerts"""
        current_time = time.time()
        
        if current_time - self.last_eye_tts_time > self.eye_tts_cooldown:
            if eye_results['drowsiness_detected']:
                self.speak_async("Warning! Eyes have been closed for more than 3 seconds. Please stay alert!")
                self.last_eye_tts_time = current_time
                return True
        return False
    
    def handle_phone_alert(self, phone_results):
        """Handle phone-related TTS alerts"""
        current_time = time.time()
        
        if current_time - self.last_phone_tts_time > self.phone_tts_cooldown:
            if phone_results['continuous_detection']:
                self.speak_async("Warning! Mobile phone detected for more than 3 seconds. Please put it away and focus ahead.")
                self.last_phone_tts_time = current_time
                return True
            elif phone_results['removal_ready']:
                self.speak_async("Good! Mobile phone removed. Keep focusing ahead.")
                self.last_phone_tts_time = current_time
                return True
        return False
    
    def handle_gaze_alert(self, gaze_direction, gaze_duration, continuous_direction):
        """Handle gaze-related TTS alerts"""
        current_time = time.time()
        
        if current_time - self.last_gaze_tts_time > self.gaze_tts_cooldown:
            if continuous_direction and gaze_direction in ["left", "right"]:
                self.speak_async("Please keep your head straight!")
                self.last_gaze_tts_time = current_time
                return True
        return False
    
    def speak_sync(self, text):
        """Speak text synchronously"""
        with self.tts_lock:
            try:
                if self.tts_engine:
                    print(f"🔊 TTS: {text}")
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ TTS Error: {e}")
    
    def set_voice_properties(self, rate=None, volume=None, voice_index=None):
        """Set TTS voice properties"""
        if not self.tts_engine:
            return
            
        try:
            if rate is not None:
                self.tts_engine.setProperty('rate', rate)
                print(f"TTS rate set to: {rate}")
            
            if volume is not None:
                self.tts_engine.setProperty('volume', volume)
                print(f"TTS volume set to: {volume}")
            
            if voice_index is not None:
                voices = self.tts_engine.getProperty('voices')
                if voices and 0 <= voice_index < len(voices):
                    self.tts_engine.setProperty('voice', voices[voice_index].id)
                    print(f"TTS voice set to: {voices[voice_index].name}")
                    
        except Exception as e:
            print(f"❌ Error setting TTS properties: {e}")
    
    def get_available_voices(self):
        """Get list of available voices"""
        if not self.tts_engine:
            return []
            
        try:
            voices = self.tts_engine.getProperty('voices')
            voice_info = []
            for i, voice in enumerate(voices):
                voice_info.append({
                    'index': i,
                    'id': voice.id,
                    'name': voice.name,
                    'language': getattr(voice, 'languages', ['Unknown'])
                })
            return voice_info
        except Exception as e:
            print(f"❌ Error getting voices: {e}")
            return []
    
    def test_tts(self):
        """Test TTS functionality"""
        self.speak_async("Text to speech system is working correctly")
    
    def stop_tts(self):
        """Stop current TTS"""
        try:
            if self.tts_engine:
                self.tts_engine.stop()
        except Exception as e:
            print(f"❌ Error stopping TTS: {e}")
    
    def is_available(self):
        """Check if TTS is available"""
        return self.tts_engine is not None
