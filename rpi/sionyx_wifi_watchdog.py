#!/usr/bin/env python3
"""
SiOnyx WiFi Watchdog - Use NetworkManager properly with DHCP
"""

import subprocess
import time
import sys

CAMERA_SSID = "SiOnyx-7854D5"
CAMERA_PASSWORD = "SiOnyx_cam"
CHECK_INTERVAL = 15

def is_connected():
    """Check if connected to camera SSID"""
    try:
        result = subprocess.run(['iwgetid', '-r'], 
                               capture_output=True, text=True, timeout=2)
        return result.stdout.strip() == CAMERA_SSID
    except:
        return False

def ensure_connection_profile():
    """Create/update NetworkManager connection profile"""
    
    # Check if profile exists
    result = subprocess.run(['nmcli', '-t', '-f', 'NAME', 'connection', 'show'],
                           capture_output=True, text=True)
    
    if CAMERA_SSID in result.stdout:
        print(f"[WiFi] Connection profile exists, updating...")
        # Update existing
        subprocess.run([
            'nmcli', 'connection', 'modify', CAMERA_SSID,
            'connection.autoconnect', 'yes',
            'connection.autoconnect-priority', '100',
            'connection.multi-connect', '3'
        ])
    else:
        print(f"[WiFi] Creating connection profile...")
        # Create new
        subprocess.run([
            'nmcli', 'connection', 'add',
            'type', 'wifi',
            'con-name', CAMERA_SSID,
            'ssid', CAMERA_SSID,
            'wifi-sec.key-mgmt', 'wpa-psk',
            'wifi-sec.psk', CAMERA_PASSWORD,
            '802-11-wireless.hidden', 'yes',
            'connection.autoconnect', 'yes',
            'connection.autoconnect-priority', '100',
            'connection.multi-connect', '3'
        ])
    
    print(f"[WiFi] Profile configured")

def connect_wifi():
    """Connect using NetworkManager"""
    print(f"[WiFi] Bringing up connection...")
    
    result = subprocess.run(['nmcli', 'connection', 'up', CAMERA_SSID],
                           capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        print(f"[WiFi] ✓ Connected")
        return True
    else:
        print(f"[WiFi] Connection failed")
        return False

def main():
    print("="*60)
    print("SiOnyx WiFi Watchdog - NetworkManager")
    print("="*60)
    print(f"Target: {CAMERA_SSID}")
    print()
    
    # Ensure profile exists with correct settings
    ensure_connection_profile()
    
    while True:
        try:
            if is_connected():
                print(f"[WiFi] ✓ Connected")
            else:
                print(f"[WiFi] Not connected - connecting...")
                connect_wifi()
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[WiFi] Stopped")
            sys.exit(0)
        except Exception as e:
            print(f"[WiFi] Error: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()