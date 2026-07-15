import os
import zipfile
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    zip_path = root.parent / "f1-viz-source.zip"
    
    # Exclude directories
    exclude_dirs = {
        "data", ".git", ".pytest_cache", "node_modules", 
        "venv", ".venv", "__pycache__", ".mypy_cache"
    }
    
    # Exclude specific files
    exclude_files = {
        ".env", "f1-viz-source.zip", "temp_session.py"
    }
    
    print(f"Creating zip archive at: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root_dir, dirs, files in os.walk(root):
            # Modify dirs in-place to prevent walking into excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files or file.endswith('.pyc') or file.endswith('.pyo'):
                    continue
                    
                file_path = Path(root_dir) / file
                rel_path = file_path.relative_to(root)
                
                zipf.write(file_path, rel_path)
                
    print(f"Zip file successfully created at {zip_path}!")

if __name__ == "__main__":
    main()
