# tests/test_config.py
import unittest
from src import config

class TestConfiguration(unittest.TestCase):
    """Test cases for configuration module."""

    def test_environment_variables(self):
        """Test if environment variables are properly set."""
        self.assertTrue(config.verify_environment())

    def test_directory_structure(self):
        """Test if all required directories exist."""
        for name, path in config.DIRECTORIES.items():
            self.assertTrue(path.exists(), f"Directory {name} does not exist")

    def test_hardware_configuration(self):
        """Test hardware configuration detection."""
        device = config.configure_hardware()
        self.assertIn(device, ['cpu', 'cuda'], "Invalid device type")

if __name__ == '__main__':
    unittest.main(verbosity=2)