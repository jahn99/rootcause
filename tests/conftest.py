"""Keep tests independent of a developer's local environment files."""

from app.config import Settings

Settings.model_config["env_file"] = None
