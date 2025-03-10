#!/bin/bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
gunicorn -w 4 -b 0.0.0.0:$PORT app:app