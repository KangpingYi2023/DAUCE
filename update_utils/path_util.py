import os
from pathlib import Path


def get_project_root() -> Path:
    current_file = os.path.abspath(__file__)
    current_path = Path(current_file)
    root_path = current_path.parent.parent  
    return root_path


def get_absolute_path(relative_path: str) -> Path:
    root_path = get_project_root()
    absolute_path = root_path / relative_path
    return absolute_path


def convert_path_to_linux_style(path: str) -> str:
    return str(path).replace("\\", "/")


if __name__ == "__main__":
    print("root path of project:", get_project_root())
    print("absolute path of given file:", get_absolute_path("some/relative/path/to/file.txt"))
