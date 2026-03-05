#!/bin/bash
# Generate self-signed certificate for HTTPS access

cd /Users/macbookpro/PycharmProjects/Ryden

# Generate self-signed SSL certificate if it doesn't exist
if [ ! -f "cert.pem" ]; then
    echo "Generating self-signed SSL certificate..."
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    echo "✓ Certificate generated"
fi

# Start server with SSL
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --ssl-keyfile=key.pem --ssl-certfile=cert.pem
