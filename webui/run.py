#!/usr/bin/env python3
"""
Kronos Web UI startup script
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import plotly
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def install_dependencies():
    """Install dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installation completed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Dependencies installation failed")
        return False

def main():
    """Main function"""
    print("🚀 Starting Kronos Web UI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nAuto-install dependencies? (y/n): ", end="")
        if input().lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("Please manually install dependencies and retry")
            return
    
    # Check model availability
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos model library available")
        model_available = True
    except ImportError:
        print("⚠️  Kronos model library not available, will use simulated prediction")
        model_available = False
    
    # Start Flask application
    print("\n🌐 Starting Web server (kronos_qlib data layer)...")

    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'

    # Start server
    try:
        from app import app, _ENV_ERROR
        if _ENV_ERROR:
            print("⚠️  " + _ENV_ERROR)
        print("✅ Web server started successfully!")
        print(f"🌐 Access URL: http://localhost:7070")
        print("💡 Tip: Press Ctrl+C to stop server")

        # Auto-open browser（无头环境无浏览器，忽略异常）
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:7070')
        except Exception:
            pass

        # 计划陷阱 1：qlib 数据层非线程安全——必须单线程跑；debug reloader 会
        # 双进程 init qlib，故 debug=False（与 app.py 的 __main__ 同口径）。
        app.run(debug=False, host='0.0.0.0', port=7070, threaded=False)

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        print("Please check if port 7070 is occupied")

if __name__ == "__main__":
    main()
