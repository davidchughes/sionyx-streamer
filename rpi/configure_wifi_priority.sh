#!/bin/bash
# Configure NetworkManager to prioritize SiOnyx camera WiFi

CAMERA_SSID="SiOnyx-7854D5"

echo "Configuring NetworkManager for $CAMERA_SSID..."

# Set camera WiFi to auto-connect and higher priority
nmcli connection modify "$CAMERA_SSID" \
    connection.autoconnect yes \
    connection.autoconnect-priority 100

# Set ethernet to lower priority (default is 0)
ETH_CONN=$(nmcli -t -f NAME,TYPE connection show | grep ethernet | cut -d: -f1 | head -1)
if [ -n "$ETH_CONN" ]; then
    nmcli connection modify "$ETH_CONN" connection.autoconnect-priority -10
    echo "Set ethernet priority lower than WiFi"
fi

# Enable both connections simultaneously
nmcli connection modify "$CAMERA_SSID" connection.multi-connect 3

echo "Done! Camera WiFi will now auto-connect even with ethernet plugged in"
echo
echo "To verify:"
echo "  nmcli connection show \"$CAMERA_SSID\" | grep autoconnect"
