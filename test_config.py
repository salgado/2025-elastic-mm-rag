#!/usr/bin/env python3
# test_config.py

from src import config

def test_configuration():
    print("🔍 Testing configuration...")
    
    # Test environment
    print("\n1. Testing environment variables:")
    if config.verify_environment():
        print("✅ Environment variables are properly set")
    else:
        print("❌ Some environment variables are missing")
    
    # Test directories
    print("\n2. Testing directory structure:")
    for name, path in config.DIRECTORIES.items():
        if path.exists():
            print(f"✅ {name} directory exists at {path}")
        else:
            print(f"❌ {name} directory is missing")
    
    # Test hardware configuration
    print("\n3. Testing hardware configuration:")
    device = config.configure_hardware()
    print(f"✅ Using device: {device}")

if __name__ == "__main__":
    test_configuration()