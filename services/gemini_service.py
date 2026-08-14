import os
import time
import tempfile
import datetime
from google import genai

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("Server Configuration Error: GEMINI_API_KEY environment variable is missing.")
    return genai.Client(api_key=api_key)

def cleanup_old_files(client):
    """
    Deletes files older than 2 hours from Gemini storage to prevent hitting 403 / storage quota limits.
    """
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        for f in client.files.list():
            if hasattr(f, 'create_time') and f.create_time:
                create_time = f.create_time
                if create_time.tzinfo is None:
                    create_time = create_time.replace(tzinfo=datetime.timezone.utc)
                
                age = now - create_time
                if age.total_seconds() > 7200:
                    try:
                        client.files.delete(name=f.name)
                    except Exception:
                        pass
    except Exception as e:
        print(f"Cleanup non-fatal error: {e}")

def upload_and_wait_active(client, upload_file):
    try:
        # Free up storage before performing new upload
        cleanup_old_files(client)
    except Exception as e:
        if "403" in str(e) or "PERMISSION_DENIED" in str(e):
             raise Exception("API Key Error (403): Your Gemini API Key lacks permission. Ensure Google Cloud Console 'Application Restrictions' are set to 'None' for backend usage.")
        print(f"Cleanup non-fatal error: {e}")
        
    # Reset stream pointer to start
    upload_file.seek(0)
    
    temp_dir = "/tmp" if os.path.exists("/tmp") else tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, upload_file.filename)
    
    upload_file.save(temp_path)
    
    try:
        gemini_file = client.files.upload(file=temp_path)
        
        while True:
            file_info = client.files.get(name=gemini_file.name)
            state_str = file_info.state.name if hasattr(file_info.state, 'name') else str(file_info.state)
            
            if state_str == "ACTIVE":
                break
            elif state_str == "FAILED":
                raise Exception("Gemini File processing failed on Google's servers.")
                
            time.sleep(2)
            
        return gemini_file.name
    except Exception as e:
        if "403" in str(e) or "PERMISSION_DENIED" in str(e):
            raise Exception("API Key Error (403): Your Gemini API Key lacks permission. Please remove 'Website' or 'HTTP Referrer' restrictions in the Google Cloud Console. Backend Python servers cannot use web-restricted keys.")
        raise e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)