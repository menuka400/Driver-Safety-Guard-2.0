class MainTracker:
    def __init__(self, eye_detector, phone_detector, tts_manager=None):
        self.eye_detector = eye_detector
        self.phone_detector = phone_detector
        self.tts_manager = tts_manager

    def process_frame(self, frame):
        """Process a single frame and return the annotated frame"""
        try:
            # Get eye state and gaze direction
            eye_results = self.eye_detector.detect(frame)
            eye_state = eye_results['eye_state']
            gaze_direction = eye_results['gaze_direction']
            gaze_duration = eye_results['gaze_duration']
            continuous_direction = eye_results['continuous_direction']
            
            # Get phone detection results
            phone_results = self.phone_detector.detect(frame)
            phone_detected = phone_results['phone_detected']
            phone_duration = phone_results['duration']
            continuous_phone = phone_results['continuous_detection']
            
            # Handle TTS alerts
            if self.tts_manager:
                # Handle eye state alerts
                if eye_results['drowsiness_detected']:
                    self.tts_manager.handle_eye_alert(eye_state, eye_results['closure_duration'])
                
                # Handle gaze direction alerts
                self.tts_manager.handle_gaze_alert(gaze_direction, gaze_duration, continuous_direction)
                
                # Handle phone alerts
                if continuous_phone:
                    self.tts_manager.handle_phone_alert(phone_detected, phone_duration)
            
            # Draw annotations
            self.draw_annotations(frame, eye_results, phone_results)
            
            return frame
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame 