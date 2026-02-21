#!/usr/bin/env python3
"""
SiOnyx WiFi Watchdog
Explicitly reconnects to the SiOnyx camera's hidden WiFi every 15 seconds.
Does not rely on NetworkManager autoconnect -- Python drives all reconnection.
"""

import subprocess
import time
import sys

CAMERA_SSID = "SiOnyx-7854D5"
CAMERA_PASSWORD = "SiOnyx_cam"
INTERFACE = "wlan0"
PROFILE_NAME = "sionyx-camera"
CHECK_INTERVAL = 15


def run(cmd, timeout=15):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        print(f"[WiFi] Timeout: {' '.join(cmd)}")
        return -1, "", "timeout"
    except Exception as e:
        print(f"[WiFi] Error: {e}")
        return -1, "", str(e)


def ensure_interface_up():
    run(['rfkill', 'unblock', 'wifi'])
    run(['ip', 'link', 'set', INTERFACE, 'up'])
    run(['nmcli', 'device', 'set', INTERFACE, 'managed', 'yes'])


def setup_profile():
    print(f"[WiFi] Setting up connection profile for \"{CAMERA_SSID}\"...")

    # Always delete and recreate -- never trust stale profile state
    for name in [PROFILE_NAME, CAMERA_SSID]:
        run(['nmcli', 'connection', 'delete', name])

    rc, out, err = run([
        'nmcli', 'connection', 'add',
        'type', 'wifi',
        'ifname', INTERFACE,
        'con-name', PROFILE_NAME,
        'ssid', CAMERA_SSID,
        '802-11-wireless.hidden', 'yes',
        'wifi-sec.key-mgmt', 'wpa-psk',
        'wifi-sec.psk', CAMERA_PASSWORD,
        'ipv4.method', 'auto',
        'ipv4.never-default', 'yes',   # never route internet traffic via camera
        'ipv6.method', 'disabled',
        'connection.autoconnect', 'no', # Python drives reconnect, not NM
    ])

    if rc == 0:
        print(f"[WiFi] Profile created")
        time.sleep(2)  # let NM settle before attempting connection
        return True
    else:
        print(f"[WiFi] Profile creation failed: {err}")
        return False


def is_connected():
    rc, out, err = run(['iwgetid', '-r'])
    return rc == 0 and out == CAMERA_SSID


def connect():
    rc, out, err = run(['nmcli', 'connection', 'up', PROFILE_NAME], timeout=20)
    if rc == 0:
        print(f"[WiFi] Connected to {CAMERA_SSID}")
        return True
    else:
        print(f"[WiFi] Connect failed: {err}")
        return False


def main():
    print("=" * 60)
    print("SiOnyx WiFi Watchdog")
    print("=" * 60)
    print(f"Target SSID:  {CAMERA_SSID}")
    print(f"Interface:    {INTERFACE}")
    print(f"Check every:  {CHECK_INTERVAL} seconds")
    print()

    ensure_interface_up()
    setup_profile()

    consecutive_failures = 0

    while True:
        try:
            if is_connected():
                print(f"[WiFi] Connected")
                consecutive_failures = 0
            else:
                print(f"[WiFi] Not connected -- reconnecting...")
                ensure_interface_up()
                success = connect()

                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    print(f"[WiFi] Failed ({consecutive_failures} in a row)")
                    if consecutive_failures % 5 == 0:
                        print(f"[WiFi] Rebuilding profile after {consecutive_failures} failures...")
                        setup_profile()

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n[WiFi] Stopped")
            sys.exit(0)
        except Exception as e:
            print(f"[WiFi] Unexpected error: {e}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()