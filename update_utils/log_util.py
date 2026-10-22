from pathlib import Path


def append_to_file(file_path: Path, content: str):
    print(content)
    with open(file_path, 'a') as f:
        f.write(content)
        f.flush()
