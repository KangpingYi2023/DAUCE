import copy
import sys
from pathlib import Path
import subprocess
from typing import Dict, List, Any
from abc import ABC, abstractmethod


class BaseScriptRunner(ABC):
    """
    A base class for script runners.
    """

    def __init__(
            self,
            script_path: Path,
            args: Dict[str, Any],
            output_file_path: Path = None
    ):
        """
        Initialize the BaseScriptRunner with a script path and arguments.

        :param script_path: Path to the script to be run.
        :param args: A dictionary of argument names and values.
        :param output_file_path: Path to the log file.
        """
        self.script_path = script_path
        self.args = copy.deepcopy(args)
        self.log_file_path = output_file_path

    @abstractmethod
    def run_script(self):
        """
        Abstract method to execute the script. This method should be implemented in derived classes.
        """
        pass


class PythonScriptRunner(BaseScriptRunner):
    """
    A class to run a python script with given arguments using subprocess.
    """

    def _generate_python_script(self) -> List[str]:
        # Use sys.executable rather than a bare "python" so the subprocess always
        # runs in the same interpreter/conda env as the orchestrator. A bare
        # "python" silently falls back to whatever is first on PATH (e.g. the
        # system python without torch) when the env is not activated.
        cmd = [sys.executable, str(self.script_path)]
        for k, v in self.args.items():
            if v is None:
                cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}={v}")
        return cmd

    def run_script(self):
        """
        Execute the python script with the specified arguments using "subprocess.run".

        The child's return code is always checked: a non-zero exit (i.e. the
        child raised/crashed) is re-raised as CalledProcessError instead of being
        silently ignored. Otherwise a crashed data-update child would leave the
        drift state file stale and the experiment would keep running on wrong data.
        """
        # Construct the command with arguments
        cmd = self._generate_python_script()
        print(f"PythonScriptRunner.run_script -> Running command: {cmd}")

        # Execute the command
        if self.log_file_path:
            # redirect output to the file
            with open(self.log_file_path, 'a') as output_file:
                output_file.write(f"Going to run: {cmd}\n")
                output_file.flush()
                result = subprocess.run(cmd, stdout=output_file, stderr=output_file)
                # output_file.write("\n\n\n")
        else:
            result = subprocess.run(cmd)

        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result
