#!/bin/bash

set -e

SVG_FILE="new_gui/images/main_icon.svg"
ICONSET_DIR="icon.iconset"
ICNS_FILE="app_icon.icns"

echo "Converting SVG to ICNS..."

rm -rf "$ICONSET_DIR"
mkdir "$ICONSET_DIR"

sizes=(16 32 64 128 256 512 1024)

for size in "${sizes[@]}"; do
    echo "Generating ${size}x${size}..."

    if command -v rsvg-convert &> /dev/null; then
        rsvg-convert -w $size -h $size "$SVG_FILE" > "$ICONSET_DIR/icon_${size}x${size}.png"
        if [ $size -ne 1024 ]; then
            size2=$((size * 2))
            rsvg-convert -w $size2 -h $size2 "$SVG_FILE" > "$ICONSET_DIR/icon_${size}x${size}@2x.png"
        fi
    elif command -v qlmanage &> /dev/null; then
        qlmanage -t -s $size -o "$ICONSET_DIR" "$SVG_FILE" &> /dev/null
        mv "$ICONSET_DIR/$(basename $SVG_FILE).png" "$ICONSET_DIR/icon_${size}x${size}.png"
        if [ $size -ne 1024 ]; then
            size2=$((size * 2))
            qlmanage -t -s $size2 -o "$ICONSET_DIR" "$SVG_FILE" &> /dev/null
            mv "$ICONSET_DIR/$(basename $SVG_FILE).png" "$ICONSET_DIR/icon_${size}x${size}@2x.png"
        fi
    elif command -v sips &> /dev/null; then
        sips -s format png -z $size $size "$SVG_FILE" --out "$ICONSET_DIR/icon_${size}x${size}.png" &> /dev/null
        if [ $size -ne 1024 ]; then
            size2=$((size * 2))
            sips -s format png -z $size2 $size2 "$SVG_FILE" --out "$ICONSET_DIR/icon_${size}x${size}@2x.png" &> /dev/null
        fi
    else
        echo "Error: No suitable tool found for SVG conversion"
        echo "Install librsvg with: brew install librsvg"
        exit 1
    fi
done

iconutil -c icns "$ICONSET_DIR" -o "$ICNS_FILE"

rm -rf "$ICONSET_DIR"

echo "Icon created: $ICNS_FILE"
