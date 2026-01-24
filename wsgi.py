"""WSGI entrypoint for production deployment.

Usage with uWSGI:
    uwsgi --http :5000 --wsgi-file wsgi.py --callable app --processes 4 --threads 2

Usage with Gunicorn:
    gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app

Environment variables:
    API_AUTH_ENABLED=true     Enable JWT authentication
    JWT_SECRET_KEY=<secret>   Secret key for JWT signing (required in production)
    API_USERNAME=<username>   Username for auth (default: admin)
    API_PASSWORD=<password>   Password for auth (default: changeme)
"""

from api import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
