# Face_Antispoofing_System
export QT_QPA_PLATFORM=xcb

streamlit run streamlit_app.py

Use this:

I converted my raw Python detection script into a clean Streamlit web application.
The app opens in a browser with a simple “Start Liveness Check” button.
It reads the webcam feed inside the page, processes frames, and uses the trained CNN model to classify faces as Real or Spoof.
This makes the system user-friendly, professional, and ready for integration into real authentication workflows.