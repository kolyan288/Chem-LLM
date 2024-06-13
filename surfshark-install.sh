#!/bin/sh
PATH="/bin:/usr/bin:/sbin:/usr/sbin"
_ARCHS=$(dpkg --print-architecture && dpkg --print-foreign-architectures)

if ! echo "$_ARCHS" | grep -E 'arm64|amd64' 1>/dev/null 2>/dev/null; then
    echo "Unsupported architecture. $( echo "$_ARCHS" | tr '\n' ' ')"
    exit 1
fi

SUDO="sudo"
if [ "$(id -u)" = "0" ]; then
    SUDO=
fi
sudo -k

echo "This script requires superuser access to install apt package."

set -ex

$SUDO apt-get remove -y surfshark surfshark-release || true

$SUDO tee /etc/apt/trusted.gpg.d/surfshark.asc << PGP_KEY
-----BEGIN PGP PUBLIC KEY BLOCK-----

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
=bVHI
-----END PGP PUBLIC KEY BLOCK-----
PGP_KEY
echo "deb https://ocean.surfshark.com/debian stretch main" | $SUDO tee /etc/apt/sources.list.d/surfshark.list
$SUDO apt-get update
$SUDO apt-get install -y surfshark

set +x

echo ""
echo "Surfshark was successfully installed."
echo ""
### END OF SCRIPT