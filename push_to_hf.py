"""
Push to HuggingFace Spaces Script
================================

Usage:
    python push_to_hf.py
"""

import os
import shutil
from pathlib import Path

def push_to_hf():
    # Get HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your HuggingFace token (from https://huggingface.co/settings/tokens): ").strip()
    
    if not token:
        print("Error: No token provided!")
        return
    
    # Save token
    os.environ["HF_TOKEN"] = token
    
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        from huggingface_hub.utils import HFValidationError
        
        api = HfApi(token=token)
        
        # Get username
        whoami = api.whoami()
        username = whoami.get("name")
        print(f"Logged in as: {username}")
        
        # Create or get space repo
        repo_id = f"{username}/Openenv"
        space_id = f"{username}/Openenv"
        
        print(f"\nCreating/updating Space: {space_id}")
        
        try:
            create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="gradio",
                space_hardware="t4-small",
                exist_ok=True,
                token=token
            )
            print("Space repository created/verified!")
        except Exception as e:
            print(f"Note: {e}")
        
        # Source folder
        source_folder = Path(r"C:\Users\sanin\OneDrive\Desktop\OpenEnv_HF_Space")
        
        if not source_folder.exists():
            print(f"Error: Source folder not found: {source_folder}")
            return
        
        # Upload files
        print(f"\nUploading files from: {source_folder}")
        
        # List files to upload
        files_to_upload = []
        for root, dirs, files in os.walk(source_folder):
            # Skip .git and __pycache__
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.gitignore']]
            for file in files:
                if file.endswith('.pyc'):
                    continue
                file_path = Path(root) / file
                files_to_upload.append(file_path)
        
        print(f"Files to upload: {len(files_to_upload)}")
        
        # Upload using the API
        for file_path in files_to_upload:
            rel_path = file_path.relative_to(source_folder)
            print(f"  Uploading: {rel_path}")
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=str(rel_path),
                    repo_id=space_id,
                    repo_type="space",
                    commit_message="Deploy SmartWarehouse environment",
                )
            except Exception as e:
                print(f"    Error uploading {rel_path}: {e}")
        
        print(f"\n{'='*60}")
        print(f"SUCCESS!")
        print(f"Your Space is live at:")
        print(f"https://huggingface.co/spaces/{space_id}")
        print(f"{'='*60}")
        
    except ImportError as e:
        print(f"Error: Could not import huggingface_hub: {e}")
        print("\nPlease run: pip install huggingface_hub")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    push_to_hf()
    input("\nPress Enter to exit...")
