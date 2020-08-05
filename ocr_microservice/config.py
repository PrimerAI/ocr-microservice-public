"""Define the configuration file for this Application."""
from primer_micro_utils.config import Config


class OCRConfig(Config):
    """Override base Config object and provide password configuration options."""

    result_mapping = None
    name_mapping_override = None

    def get_file_upload_path(self):
        """Returns a Dict of the name override file."""
        return self.get_setting("Config", "file_upload_path", "/code/", "FILE_UPLOAD_PATH")

config = OCRConfig()