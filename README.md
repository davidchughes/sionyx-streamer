# SiOnyx Streamer

Low-latency network streaming and remote control for SiOnyx Aurora night vision cameras.

## What It Does

Turns a Raspberry Pi 5 into a streaming bridge for SiOnyx Aurora cameras, enabling remote viewing and control from any browser or Python client on your network.

**Problem it solves:** SiOnyx Aurora cameras have WiFi but only support one direct connection through their mobile app. This project creates a network bridge so multiple devices can view and control the camera simultaneously over your existing network infrastructure.

## Features

- **Sub-200ms latency** MJPEG streaming at 15fps (640x360)
- **Remote control** - zoom, EIS, night modes, recording, photo capture
- **Multi-client** - Multiple viewers can watch simultaneously
- **Touch UI** - Responsive web interface for desktop and mobile
- **Python client** - For custom applications and integrations
- **Auto-recovery** - Automatic reconnection if camera or WiFi drops

## Architecture

```
SiOnyx Camera (WiFi) ←→ RPi5 (wlan0) ←→ RPi5 (eth0) ←→ Your Network
                                ↓
                         HTTP Server (port 8080)
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
            Browser Clients          Python Client
```

**Components:**
- **Camera Session** - Manages SiOnyx API communication
- **UDP Receiver** - Captures MJPEG stream from camera
- **HTTP Server** - Serves video and control endpoints
- **WiFi Watchdog** - Maintains camera WiFi connection alongside ethernet
- **Web UI** - Touch-friendly remote control interface

## Requirements

**Hardware:**
- Raspberry Pi 5 (4GB+ recommended)
- SiOnyx Aurora camera (tested with Aurora Pro)
- Ethernet connection for RPi5
- 2.4/5GHz WiFi capability on RPi5

**Software:**
- Raspberry Pi OS (64-bit, Bookworm or later)
- Python 3.9+
- NetworkManager

## Quick Start

1. **Install on RPi5:**
```bash
cd /path/to/sionyx-streamer
sudo ./install_rpi.sh
sudo systemctl enable sionyx-server sionyx-wifi
sudo systemctl start sionyx-server sionyx-wifi
```

2. **Access from browser:**
```
http://sionyx.local:8080/
```

3. **Or use Python client:**
```bash
python windows_client.py
```

## Network Configuration

The system maintains two simultaneous network connections:
- **wlan0** - Connected to SiOnyx camera WiFi (hidden SSID, DHCP assigns 192.168.0.128)
- **eth0** - Connected to your local network for client access

The WiFi watchdog ensures the camera connection stays active even when ethernet provides internet.

## Controls

**Web Interface:**
- Record/Stop toggle
- Photo capture
- Electronic Image Stabilization (EIS)
- Digital zoom (1.0x - 3.0x)
- Night modes (Color, Green, Grayscale)
- Fullscreen toggle

**Keyboard shortcuts:**
- Space - Photo
- R - Record toggle
- E - EIS toggle
- Q - Reload page

## Performance

- **Latency:** <200ms end-to-end (camera to display)
- **Frame rate:** 30fps display, 15fps camera source
- **Resolution:** 640x360 (camera native)
- **Bandwidth:** ~1-2 Mbps per client

## Troubleshooting

**Camera not connecting:**
```bash
sudo journalctl -u sionyx-wifi -f
sudo systemctl restart sionyx-wifi
```

**Stream not showing:**
```bash
sudo journalctl -u sionyx-server -f
sudo systemctl restart sionyx-server
```

**Check WiFi connection:**
```bash
iwgetid -r  # Should show: SiOnyx-7854D5
ip addr show wlan0  # Should show: 192.168.0.128
```

## Acknowledgments

Reverse-engineered SiOnyx Aurora camera protocol through network analysis and experimentation.
