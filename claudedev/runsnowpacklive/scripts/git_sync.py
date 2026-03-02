# snowpack_steiermark/scripts/git_sync.py
"""
Automatic Git commit and push of SNOWPACK simulation results.
Uses GitPython to stage PRO files, AVAPRO output, and state.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import git
    _GIT_AVAILABLE = True
except ImportError:
    _GIT_AVAILABLE = False
    logger.warning("GitPython not installed — git_sync will be a no-op.")


class GitSync:
    """Commits and pushes SNOWPACK results to the remote Git repository."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.repo_path = Path(config["git"]["repo_path"]).resolve()
        self.branch = config["git"]["branch"]
        self.msg_template = config["git"].get(
            "commit_message_template", "SNOWPACK update {station} {date}"
        )
        self.station_name = config["station"]["name"]
        self._repo: Optional[object] = None

        if not _GIT_AVAILABLE:
            logger.warning("GitPython unavailable — git operations disabled.")
            return

        try:
            self._repo = git.Repo(str(self.repo_path), search_parent_directories=True)
            logger.info("Opened git repo at: %s", self._repo.working_dir)
        except Exception as exc:
            logger.warning("Could not open git repo at %s: %s", self.repo_path, exc)

    def is_repo(self) -> bool:
        """Return True if a valid Git repository was opened."""
        return self._repo is not None

    def get_changed_files(self) -> list[str]:
        """
        Return a list of modified and untracked file paths relative to repo root.

        Returns
        -------
        list[str]
            File paths that differ from HEAD or are untracked.
        """
        if not self.is_repo():
            return []
        try:
            repo = self._repo
            changed = [item.a_path for item in repo.index.diff(None)]
            untracked = list(repo.untracked_files)
            return changed + untracked
        except Exception as exc:
            logger.warning("Could not list changed files: %s", exc)
            return []

    def push(self, message: Optional[str] = None) -> tuple[bool, str]:
        """
        Stage relevant output files, commit, and push to remote.

        Stages:
          - data/pro/  (PRO simulation output)
          - data/avapro_output/  (classification CSV)
          - state/last_download.json

        Parameters
        ----------
        message : str, optional
            Commit message.  Defaults to the template in config.yaml.

        Returns
        -------
        tuple[bool, str]
            (success, output_message).
        """
        if not _GIT_AVAILABLE:
            return False, "GitPython not installed"
        if not self.is_repo():
            return False, "no repo"

        repo = self._repo
        repo_root = Path(repo.working_dir)

        # Resolve absolute paths for staging
        project_root = Path(__file__).resolve().parent.parent
        pro_dir = (project_root / self.config["paths"]["data"] / "pro").resolve()
        avapro_dir = (project_root / self.config["paths"]["data"] / "avapro_output").resolve()
        state_file = (project_root / self.config["paths"]["state"] / "last_download.json").resolve()

        paths_to_stage = []
        for p in [pro_dir, avapro_dir, state_file]:
            if p.exists():
                paths_to_stage.append(str(p))

        if not paths_to_stage:
            logger.info("No output files found to stage.")
            return True, "nothing to commit"

        try:
            repo.index.add(paths_to_stage)
        except Exception as exc:
            logger.warning("git add failed: %s", exc)
            return False, f"git add failed: {exc}"

        # Check if anything is actually staged
        if not repo.index.diff("HEAD") and not repo.index.diff(None):
            logger.info("Nothing to commit after staging.")
            return True, "nothing to commit"

        if message is None:
            now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            message = self.msg_template.format(
                station=self.station_name, date=now_str
            )

        try:
            commit = repo.index.commit(message)
            logger.info("Committed: %s", commit.hexsha[:8])
        except Exception as exc:
            logger.error("git commit failed: %s", exc)
            return False, f"git commit failed: {exc}"

        try:
            origin = repo.remote("origin")
            push_info = origin.push(self.branch)
            push_summary = str(push_info[0].summary) if push_info else "pushed"
            logger.info("Pushed to origin/%s: %s", self.branch, push_summary)
            return True, f"Committed {commit.hexsha[:8]} and pushed: {push_summary}"
        except Exception as exc:
            logger.error("git push failed: %s", exc)
            return False, f"git push failed: {exc}"


def git_push(config: dict, message: Optional[str] = None) -> tuple[bool, str]:
    """
    Commit and push SNOWPACK results to the remote repository.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    message : str, optional
        Custom commit message.

    Returns
    -------
    tuple[bool, str]
        (success, output_message).
    """
    sync = GitSync(config)
    return sync.push(message=message)
