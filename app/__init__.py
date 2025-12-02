from flask import Flask
import os

def create_app():
    """Factory function to initialize the Flask app."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.getcwd(), 'templates'),
        static_folder=os.path.join(os.getcwd(), 'static')
    )

    # Configuration
    app.config['SECRET_KEY'] = 'APT'
    app.config['SESSION_DATA_DIR'] = os.path.join(os.getcwd(), 'data')
    app.config['STATIC_SESSIONS_DIR'] = os.path.join(os.getcwd(), 'static', 'sessions')

    # Ensure necessary directories exist
    os.makedirs(app.config['SESSION_DATA_DIR'], exist_ok=True)
    os.makedirs(app.config['STATIC_SESSIONS_DIR'], exist_ok=True)

    # Register Blueprints (for routes and modularity)
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app