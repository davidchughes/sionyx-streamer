#!/bin/bash
# ==============================================================================
#  SiOnyx Streamer -- RPi Installer
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/camera_config.txt"
SERVICE_NAME="sionyx-streamer"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON_BIN="$(which python3)"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

print_header() {
    echo ""
    echo "============================================================"
    echo "  SiOnyx Streamer -- RPi Installer"
    echo "============================================================"
    echo ""
}

print_step() {
    echo ""
    echo "Step $1 of 4: $2"
    echo "------------------------------------------------------------"
}

print_ok() {
    echo "  [OK] $1"
}

print_warn() {
    echo "  [!!] $1"
}

print_info() {
    echo "  [i]  $1"
}

# ------------------------------------------------------------------------------
# Check running as root
# ------------------------------------------------------------------------------

if [ "$EUID" -ne 0 ]; then
    echo ""
    echo "  This installer needs to run as root."
    echo "  Please re-run with: sudo bash install.sh"
    echo ""
    exit 1
fi

print_header

# ==============================================================================
# STEP 1: Camera WiFi SSID
# ==============================================================================

print_step 1 "Camera WiFi"

echo ""
echo "  Put your SiOnyx camera into WiFi mode so the SSID is visible"
echo "  on the camera's screen. It will look something like:"
echo ""
echo "      SiOnyx-7854D5"
echo ""
echo "  The camera must be displaying this name -- it is a hidden"
echo "  network and will not appear in a normal WiFi scan. If you"
echo "  cannot see the SSID on the camera screen, the RPi will not"
echo "  be able to connect."
echo ""

# Check if already configured
EXISTING_SSID=""
if [ -f "$CONFIG_FILE" ]; then
    EXISTING_SSID=$(grep -E '^\s*CAMERA_SSID\s*=' "$CONFIG_FILE" | sed 's/.*=\s*//' | tr -d '[:space:]')
fi

if [ -n "$EXISTING_SSID" ]; then
    echo "  Current SSID in camera_config.txt: $EXISTING_SSID"
    read -rp "  Press Enter to keep it, or type a new SSID: " INPUT_SSID
    if [ -z "$INPUT_SSID" ]; then
        CAMERA_SSID="$EXISTING_SSID"
    else
        CAMERA_SSID="$INPUT_SSID"
    fi
else
    read -rp "  Enter your SiOnyx camera's WiFi name: " CAMERA_SSID
    if [ -z "$CAMERA_SSID" ]; then
        echo ""
        echo "  No SSID entered. Aborting."
        exit 1
    fi
fi

# Write config file
cat > "$CONFIG_FILE" <<EOF
# WiFi name shown on your SiOnyx camera's screen when in WiFi mode.
# Put the camera into WiFi mode and look at the camera display --
# it will show something like "SiOnyx-7854D5". Enter that name below exactly.
#
# NOTE: This network is hidden and will not appear in a normal WiFi scan.
# The camera MUST be displaying the SSID on screen or the connection will fail.

CAMERA_SSID = ${CAMERA_SSID}
EOF

print_ok "Saved SSID \"${CAMERA_SSID}\" to camera_config.txt"

# ==============================================================================
# STEP 2: Network info and hostname
# ==============================================================================

print_step 2 "Network info and hostname"

echo ""
echo "  Your current wired network address:"
echo ""

# Find the best non-wlan0 IPv4 address to show the user
WIRED_IP=""
WIRED_IF=""
while IFS= read -r iface; do
    if [[ "$iface" == wlan* ]]; then
        continue
    fi
    IP=$(ip -4 addr show "$iface" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
    if [ -n "$IP" ]; then
        WIRED_IP="$IP"
        WIRED_IF="$iface"
        break
    fi
done < <(ip -o link show | awk -F': ' '{print $2}' | grep -v lo)

if [ -n "$WIRED_IP" ]; then
    echo "      ${WIRED_IF}  ->  ${WIRED_IP}   <-- write this down"
else
    echo "      No wired IP found yet. Make sure an ethernet cable is plugged in."
    echo "      You can find the IP later with: hostname -I"
    WIRED_IP="(unknown)"
fi

echo ""

# Hostname
CURRENT_HOSTNAME=$(hostname)
echo "  Current hostname: $CURRENT_HOSTNAME"
echo ""

if [ "$CURRENT_HOSTNAME" = "raspberrypi" ]; then
    print_warn "Your hostname is the default \"raspberrypi\"."
    echo "         We recommend renaming it to \"sionyx\" so you can reach"
    echo "         this device at http://sionyx.local from any browser."
    echo ""
    read -rp "  Rename hostname to \"sionyx\"? [Y/n]: " RENAME_HOST
    RENAME_HOST="${RENAME_HOST:-Y}"
    if [[ "$RENAME_HOST" =~ ^[Yy] ]]; then
        hostnamectl set-hostname sionyx
        # Update /etc/hosts too
        sed -i "s/raspberrypi/sionyx/g" /etc/hosts
        print_ok "Hostname set to \"sionyx\""
        DISPLAY_HOSTNAME="sionyx"
    else
        print_info "Keeping hostname as \"$CURRENT_HOSTNAME\""
        DISPLAY_HOSTNAME="$CURRENT_HOSTNAME"
    fi
else
    print_ok "Hostname is already \"$CURRENT_HOSTNAME\" -- no change needed"
    DISPLAY_HOSTNAME="$CURRENT_HOSTNAME"
fi

# ==============================================================================
# STEP 3: Install dependencies
# ==============================================================================

print_step 3 "Installing dependencies"

echo ""
apt-get update -qq
apt-get install -y -qq python3-pip python3-numpy network-manager

print_ok "System packages installed"

pip3 install --break-system-packages --quiet opencv-python-headless requests 2>/dev/null || \
pip3 install --quiet opencv-python-headless requests

print_ok "Python packages installed"

# ==============================================================================
# Configure WiFi via NetworkManager
# ==============================================================================

echo ""
echo "  Configuring WiFi connection for camera SSID..."

# Remove any existing connection with the same name
nmcli connection delete "sionyx-camera" 2>/dev/null || true

# Add the hidden network connection
# Camera always has IP 192.168.0.1, RPi gets 192.168.0.100
nmcli connection add \
    type wifi \
    ifname wlan0 \
    con-name "sionyx-camera" \
    ssid "$CAMERA_SSID" \
    -- \
    wifi.hidden yes \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "" \
    ipv4.method manual \
    ipv4.addresses "192.168.0.100/24" \
    ipv4.gateway "192.168.0.1" \
    ipv4.never-default yes \
    connection.autoconnect yes \
    connection.autoconnect-retries 0 2>/dev/null || true

# Try without password first (open network)
nmcli connection modify "sionyx-camera" \
    wifi-sec.key-mgmt none 2>/dev/null || true

print_ok "WiFi connection profile created for \"${CAMERA_SSID}\""
print_info "The RPi will continuously try to connect to the camera WiFi."
print_info "The wired network connection is unaffected."

# ==============================================================================
# STEP 4: systemd service
# ==============================================================================

print_step 4 "Setting up auto-start service"

echo ""

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=SiOnyx Streamer Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${PYTHON_BIN} ${SCRIPT_DIR}/rpi_server.py
Restart=always
RestartSec=5
StartLimitIntervalSec=0
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sionyx-streamer

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

print_ok "Service installed and enabled (auto-starts on boot)"
print_ok "Service started now"

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "============================================================"
echo "  All done!"
echo ""
echo "  Web interface:   http://${DISPLAY_HOSTNAME}.local:8080"
echo "  Backup address:  http://${WIRED_IP}:8080"
echo ""
echo "  If http://${DISPLAY_HOSTNAME}.local does not work on your"
echo "  router, use the backup address above instead."
echo ""
echo "  HOW IT WORKS:"
echo "    As long as this RPi is powered on, it will continuously"
echo "    look for the SiOnyx camera WiFi and connect automatically."
echo ""
echo "  IF THE CAMERA IS NOT CONNECTING:"
echo "    1. Put the camera into WiFi mode"
echo "    2. The SSID must be visible on the camera screen"
echo "    3. Clear any warnings on camera (e.g. low battery alerts)"
echo "    4. Within 15 seconds of seeing the SSID on screen,"
echo "       the RPi should connect automatically"
echo "    5. If still not working, turn the camera off and back"
echo "       to WiFi mode to reset it"
echo ""
echo "  To check server status:  sudo systemctl status sionyx-streamer"
echo "  To view logs:            sudo journalctl -u sionyx-streamer -f"
echo "============================================================"
echo ""
