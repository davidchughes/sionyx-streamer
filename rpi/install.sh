#!/bin/bash
# ==============================================================================
#  SiOnyx Streamer -- RPi Installer
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/camera_config.txt"
WATCHDOG_SCRIPT="$SCRIPT_DIR/sionyx_wifi_watchdog.py"
SERVICE_NAME="sionyx-streamer"
WIFI_SERVICE_NAME="sionyx-wifi"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
WIFI_SERVICE_FILE="/etc/systemd/system/${WIFI_SERVICE_NAME}.service"
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
    echo "Step $1 of 5: $2"
    echo "------------------------------------------------------------"
}

print_ok()   { echo "  [OK] $1"; }
print_warn() { echo "  [!!] $1"; }
print_info() { echo "  [i]  $1"; }

# ------------------------------------------------------------------------------
# Must run as root
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

print_step 1 "Camera WiFi SSID"

echo ""
echo "  Put your SiOnyx camera into WiFi mode so the SSID is visible"
echo "  on the camera's screen. It will look something like:"
echo ""
echo "      SiOnyx-7854D5"
echo ""
echo "  The camera must be displaying this name -- it is a hidden"
echo "  network and will not appear in a normal WiFi scan."
echo ""

EXISTING_SSID=""
if [ -f "$CONFIG_FILE" ]; then
    EXISTING_SSID=$(grep -E '^\s*CAMERA_SSID\s*=' "$CONFIG_FILE" | sed 's/.*=\s*//' | tr -d '[:space:]')
fi

if [ -n "$EXISTING_SSID" ]; then
    echo "  Current SSID in camera_config.txt: $EXISTING_SSID"
    read -rp "  Press Enter to keep it, or type a new SSID: " INPUT_SSID
    CAMERA_SSID="${INPUT_SSID:-$EXISTING_SSID}"
else
    read -rp "  Enter your SiOnyx camera's WiFi name: " CAMERA_SSID
    if [ -z "$CAMERA_SSID" ]; then
        echo "  No SSID entered. Aborting."
        exit 1
    fi
fi

# Warn if SSID does not look like a full SiOnyx name --
# common mistake is entering just the suffix e.g. "7854D5" instead of "SiOnyx-7854D5"
if [[ "$CAMERA_SSID" != SiOnyx-* ]]; then
    echo ""
    print_warn "\"${CAMERA_SSID}\" doesn't start with \"SiOnyx-\"."
    echo "         The full name on the camera screen looks like:"
    echo "         SiOnyx-7854D5  (including the \"SiOnyx-\" prefix)"
    echo ""
    read -rp "  Enter the full SSID (or press Enter to keep \"${CAMERA_SSID}\"): " RETRY_SSID
    if [ -n "$RETRY_SSID" ]; then
        CAMERA_SSID="$RETRY_SSID"
    fi
fi

# Write camera_config.txt
cat > "$CONFIG_FILE" <<EOF
# WiFi name shown on your SiOnyx camera's screen when in WiFi mode.
# It will look something like "SiOnyx-7854D5".
# The camera MUST be displaying this name -- it is a hidden network.
# Clear any warnings on the camera (e.g. low battery) with the Enter button.

CAMERA_SSID = ${CAMERA_SSID}
EOF

print_ok "Saved SSID \"${CAMERA_SSID}\" to camera_config.txt"

# Stamp SSID into watchdog script
if [ ! -f "$WATCHDOG_SCRIPT" ]; then
    echo "  Error: sionyx_wifi_watchdog.py not found in $SCRIPT_DIR"
    exit 1
fi
sed -i "s|^CAMERA_SSID = .*|CAMERA_SSID = \"${CAMERA_SSID}\"|" "$WATCHDOG_SCRIPT"
print_ok "SSID stamped into sionyx_wifi_watchdog.py"

# ==============================================================================
# STEP 2: Network info and hostname
# ==============================================================================

print_step 2 "Network info and hostname"

echo ""
echo "  Your current wired network address:"
echo ""

WIRED_IP=""
WIRED_IF=""
while IFS= read -r iface; do
    if [[ "$iface" == wlan* ]]; then continue; fi
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
    echo "      No wired IP found. Make sure an ethernet cable is plugged in."
    echo "      You can find the IP later with: hostname -I"
    WIRED_IP="(unknown)"
fi

echo ""

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
        print_ok "hostnamectl set to sionyx"

        # Disable cloud-init host management FIRST so it stops overwriting /etc/hosts
        # Try every known location and format
        for cfg in /etc/cloud/cloud.cfg /boot/firmware/user-data /boot/user-data; do
            if [ -f "$cfg" ]; then
                sed -i "s/manage_etc_hosts: true/manage_etc_hosts: false/g" "$cfg"
                sed -i "s/manage_etc_hosts: True/manage_etc_hosts: false/g" "$cfg"
                print_ok "Cloud-init manage_etc_hosts disabled in $cfg"
            fi
        done

        # Write /etc/hostname directly
        echo "sionyx" > /etc/hostname

        # Cloud-init drop-in that always wins:
        #   manage_etc_hosts: false  -- stop overwriting /etc/hosts
        #   preserve_hostname: true  -- stop resetting hostname on boot
        mkdir -p /etc/cloud/cloud.cfg.d/
        cat > /etc/cloud/cloud.cfg.d/99_sionyx.cfg <<CLOUDINIT
manage_etc_hosts: false
preserve_hostname: true
CLOUDINIT
        print_ok "Cloud-init override written to /etc/cloud/cloud.cfg.d/99_sionyx.cfg"

        # Update the template so if cloud-init does run it uses the right name
        for tmpl in /etc/cloud/templates/hosts.debian.tmpl                     /etc/cloud/templates/hosts.tmpl                     /usr/lib/cloud-init/templates/hosts.debian.tmpl; do
            if [ -f "$tmpl" ]; then
                sed -i "s/raspberrypi/sionyx/g" "$tmpl"
                print_ok "Cloud-init template updated: $tmpl"
            fi
        done

        # Now edit /etc/hosts directly
        sed -i "s/raspberrypi/sionyx/g" /etc/hosts
        print_ok "/etc/hosts updated"

        # Verify it took
        if grep -q "sionyx" /etc/hosts; then
            print_ok "/etc/hosts contains sionyx -- hostname change complete"
        else
            print_warn "/etc/hosts update may not have taken -- check manually"
        fi

        print_ok "Hostname set to \"sionyx\""
        print_ok "sionyx.local will resolve after reboot"
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
apt-get install -y -qq python3-pip python3-numpy network-manager wireless-tools rfkill avahi-daemon
print_ok "System packages installed"

systemctl enable avahi-daemon
systemctl restart avahi-daemon
print_ok "avahi-daemon enabled (makes hostname.local resolve on the network)"

pip3 install --break-system-packages --quiet opencv-python-headless requests 2>/dev/null || \
pip3 install --quiet opencv-python-headless requests
print_ok "Python packages installed"

# ==============================================================================
# STEP 4: WiFi setup
# ==============================================================================

print_step 4 "WiFi setup"

echo ""

# Set WiFi country code -- the radio stays soft-blocked on RPi OS images
# that were set up without a WiFi connection until this is done
raspi-config nonint do_wifi_country US
print_ok "WiFi country code set (unblocks radio on fresh RPi OS installs)"

# Unblock radio and bring interface up
rfkill unblock wifi
rfkill unblock all
ip link set wlan0 up 2>/dev/null || true
print_ok "WiFi radio unblocked"

# Stop standalone wpa_supplicant -- NetworkManager runs its own internally
# and they will fight each other if both are active
systemctl stop wpa_supplicant 2>/dev/null || true
systemctl disable wpa_supplicant 2>/dev/null || true
systemctl stop wpa_supplicant@wlan0 2>/dev/null || true
systemctl disable wpa_supplicant@wlan0 2>/dev/null || true
print_ok "Standalone wpa_supplicant disabled"

# Tell NetworkManager to manage wlan0
# RPi OS often marks it unmanaged by default
mkdir -p /etc/NetworkManager/conf.d/
cat > /etc/NetworkManager/conf.d/sionyx.conf <<EOF
[main]
plugins=keyfile

[keyfile]
unmanaged-devices=none
EOF
print_ok "NetworkManager configured to manage wlan0"

systemctl enable NetworkManager
systemctl restart NetworkManager
sleep 3
print_ok "NetworkManager restarted"

# ==============================================================================
# STEP 5: Install systemd services
# ==============================================================================

print_step 5 "Installing services"

echo ""

# WiFi watchdog service
cat > "$WIFI_SERVICE_FILE" <<EOF
[Unit]
Description=SiOnyx WiFi Watchdog
After=network.target NetworkManager.service
Wants=NetworkManager.service
StartLimitIntervalSec=0

[Service]
Type=simple
User=root
WorkingDirectory=${SCRIPT_DIR}
ExecStartPre=/usr/sbin/rfkill unblock wifi
ExecStartPre=/sbin/ip link set wlan0 up
ExecStart=${PYTHON_BIN} ${WATCHDOG_SCRIPT}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sionyx-wifi

[Install]
WantedBy=multi-user.target
EOF

# Streamer server service
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=SiOnyx Streamer Server
After=network.target ${WIFI_SERVICE_NAME}.service
Wants=${WIFI_SERVICE_NAME}.service
StartLimitIntervalSec=0

[Service]
Type=simple
User=root
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${PYTHON_BIN} ${SCRIPT_DIR}/rpi_server.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sionyx-streamer

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

systemctl enable "$WIFI_SERVICE_NAME"
systemctl restart "$WIFI_SERVICE_NAME"
print_ok "WiFi watchdog service installed and started"

systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
print_ok "Streamer server service installed and started"

# ==============================================================================
# Summary + reboot prompt
# ==============================================================================

echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  Web interface:   http://${DISPLAY_HOSTNAME}.local:8080"
echo "  Backup address:  http://${WIRED_IP}:8080"
echo ""
echo "  HOW IT WORKS:"
echo "    The WiFi watchdog explicitly connects to the camera every"
echo "    15 seconds. Put the camera into WiFi mode with the SSID"
echo "    visible on screen and it will connect within 15 seconds."
echo ""
echo "  IF THE CAMERA IS NOT CONNECTING:"
echo "    1. Put the camera into WiFi mode"
echo "    2. The SSID must be visible on the camera screen"
echo "    3. Clear any warnings (e.g. low battery) with the Enter button"
echo "    4. Wait up to 15 seconds for the RPi to connect"
echo "    5. If still not working, power cycle the camera back to WiFi mode"
echo ""
echo "  USEFUL COMMANDS:"
echo "    Check WiFi:    sudo systemctl status sionyx-wifi"
echo "    Check server:  sudo systemctl status sionyx-streamer"
echo "    WiFi logs:     sudo journalctl -u sionyx-wifi -f"
echo "    Server logs:   sudo journalctl -u sionyx-streamer -f"
echo "============================================================"
echo ""

read -rp "  Reboot now to ensure WiFi starts correctly? [Y/n]: " DO_REBOOT
DO_REBOOT="${DO_REBOOT:-Y}"
if [[ "$DO_REBOOT" =~ ^[Yy] ]]; then
    echo ""
    echo "  Rebooting in 5 seconds..."
    echo "  After reboot, access the stream at http://${DISPLAY_HOSTNAME}.local:8080"
    sleep 5
    reboot
else
    echo ""
    print_info "Skipping reboot. If WiFi does not connect, run: sudo reboot"
fi