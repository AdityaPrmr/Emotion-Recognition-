#!/bin/bash
export PORT=10000
gunicorn -w 4 -b 0.0.0.0:$PORT app:app