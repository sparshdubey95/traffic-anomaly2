#!/usr/bin/env python3
"""
Quick Start Script for AI Traffic Anomaly Detection System
This script automates the setup process
"""

import os
import sys
import subprocess
import platform

def create_directory_structure():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        ".streamlit"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✅ Created: {directory}/")
        else:
            print(f"  ℹ️  Exists: {directory}/")

def create_streamlit_config():
    """Create Streamlit configuration files"""
    print("\n⚙️ Setting up Streamlit configuration...")
    
    config_content = """[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
enableCORS = false
enableXsrfProtection = false
"""
    
    config_path = ".streamlit/config.toml"
    
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"  ✅ Created: {config_path}")
    else:
        print(f"  ℹ️  Exists: {config_path}")

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("  ❌ requirements.txt not found!")
            return False
        
        # Install dependencies
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Dependencies installed successfully")
            return True
        else:
            print(f"  ❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Installation error: {e}")
        return False

def run_installation_test():
    """Run the installation test script"""
    print("\n🧪 Running installation tests...")
    
    try:
        if os.path.exists("test_installation.py"):
            result = subprocess.run([sys.executable, "test_installation.py"], 
                                 capture_output=True, text=True)
            
            print(result.stdout)
            
            if result.returncode == 0:
                return True
            else:
                print(result.stderr)
                return False
        else:
            print("  ⚠️ Test script not found, skipping tests")
            return True
            
    except Exception as e:
        print(f"  ❌ Test error: {e}")
        return False

def show_next_steps():
    """Show next steps to user"""
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Test the application:")
    print("   streamlit run app.py")
    print()
    print("2. Login with demo credentials:")
    print("   - Admin: admin / traffic2025")
    print("   - Demo:  demo / demo123") 
    print("   - Judge: judge / panscience2025")
    print()
    print("3. Upload a test video (MP4, AVI, MOV)")
    print("4. Start analysis and review results")
    print()
    print("🔗 Application will open at: http://localhost:8501")

def main():
    """Main setup function"""
    print("🚀 AI Traffic Anomaly Detection System - Quick Setup")
    print("=" * 60)
    print("This script will set up your project automatically")
    print()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        print("Please install Python 3.8 or higher")
        sys.exit(1)
    
    # Create directories
    create_directory_structure()
    
    # Create config files
    create_streamlit_config()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        print("Please run manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests
    if not run_installation_test():
        print("\n⚠️ Some tests failed, but setup may still work")
        print("Try running the application manually")
    
    # Show next steps
    show_next_steps()
    
    print("\n🎯 Ready for PanScience Innovations Competition!")

if __name__ == "__main__":
    main()