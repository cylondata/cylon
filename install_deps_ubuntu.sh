#!/bin/bash

# shellcheck disable=SC2046
if [ $(id -u) != "0" ]; then
  echo "You must be the root" >&2
  exit 1
fi

apt-get update

apt-get -y install git
apt-get -y install build-essential
apt-get -y install g++
apt-get -y install python3-dev
apt-get -y install python3-pip
apt-get -y install maven
apt-get -y install libnuma-dev
apt-get -y install libc-dev
apt-get -y install python3-venv
apt-get -y install openmpi-bin
apt-get -y install libopenmpi-dev
apt-get -y install flex
apt-get -y install bison

apt-get -y install libssl-dev
apt-get -y install libparquet-dev
apt-get -y install libsnappy-dev
apt-get -y install libbrotli-dev
apt-get -y install liblz4-dev

apt-get -y install libzstd-dev
apt-get -y install libre2-dev
apt-get -y install libgandiva-dev
apt-get -y install libbzip2-dev
apt-get -y install bzip2-dev
apt-get -y install libbz2-dev
apt-get -y install libutf8proc-dev

## Intall arrow
apt install -y -V ca-certificates lsb-release wget
if [ $(lsb_release --codename --short) = "stretch" ]; then
  tee /etc/apt/sources.list.d/backports.list <<APT_LINE
deb http://deb.debian.org/debian $(lsb_release --codename --short)-backports main
APT_LINE
fi
wget https://apache.bintray.com/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb
apt install -y -V ./apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb
apt update
apt install -y -V libarrow-dev # For C++
apt install -y -V libarrow-glib-dev # For GLib (C)
apt install -y -V libarrow-dataset-dev # For Arrow Dataset C++
apt install -y -V libarrow-flight-dev # For Flight C++
apt install -y -V libplasma-dev # For Plasma C++
apt install -y -V libplasma-glib-dev # For Plasma GLib (C)
apt install -y -V libgandiva-dev # For Gandiva C++
apt install -y -V libgandiva-glib-dev # For Gandiva GLib (C)
apt install -y -V libparquet-dev # For Apache Parquet C++
apt install -y -V libparquet-glib-dev # For Apache Parquet GLib (C)

