#!/bin/bash
# SiOnyx Aurora RPi5 Server Installation Script

set -e

echo "================================"
echo "SiOnyx Aurora Server Installer"
echo "================================"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
   echo "Please run as root (sudo ./install_rpi.sh)"
   exit 1
fi

echo "NOTE: Please ensure WiFi is connected to SiOnyx-7854D5 before installing"
echo "Press Enter to continue or Ctrl+C to cancel..."
read

echo "[1/5] Installing dependencies..."
apt update
apt install -y python3-pip
pip3 install requests --break-system-packages
pip3 install opencv-python --break-system-packages

echo "[2/5] Creating directories..."
mkdir -p /opt/sionyx
mkdir -p /var/log/sionyx

echo "[3/5] Installing server script..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ln -sf "$SCRIPT_DIR/rpi_server.py" /opt/sionyx/rpi_server.py
chmod +x /opt/sionyx/rpi_server.py

echo "[4/5] Installing web interface..."
ln -sf "$SCRIPT_DIR/index.html" /opt/sionyx/index.html
ln -sf "$SCRIPT_DIR/viewer.js" /opt/sionyx/viewer.js
ln -sf "$SCRIPT_DIR/NoSleep.min.js" /opt/sionyx/NoSleep.min.js

echo "[5/5] Installing WiFi watchdog..."
ln -sf "$SCRIPT_DIR/sionyx_wifi_watchdog.py" /opt/sionyx/sionyx_wifi_watchdog.py
chmod +x /opt/sionyx/sionyx_wifi_watchdog.py
cp sionyx-wifi.service /etc/systemd/system/

echo "[6/6] Installing systemd services..."
cp sionyx-server.service /etc/systemd/system/
systemctl daemon-reload

echo
echo "================================"
echo "Installation Complete!"
echo "================================"
echo
echo "To enable WiFi watchdog (keeps camera WiFi connected):"
echo "  sudo systemctl enable sionyx-wifi"
echo "  sudo systemctl start sionyx-wifi"
echo
echo "To enable server auto-start on boot:"
echo "  sudo systemctl enable sionyx-server"
echo
echo "To start the server now:"
echo "  sudo systemctl start sionyx-server"
echo
echo "To check status:"
echo "  sudo systemctl status sionyx-server"
echo "  sudo systemctl status sionyx-wifi"
echo
echo "To view logs:"
echo "  sudo journalctl -u sionyx-server -f"
echo "  sudo journalctl -u sionyx-wifi -f"
echo
echo "Control Port: 5000"
echo "Stream Port: 8080"
echo