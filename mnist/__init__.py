import os
from flask import Flask
from flask_mongoengine import MongoEngine

# export FLASK_APP = mnist FLASK_ENV = development; flask run

db = MongoEngine()


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    ''' load configs '''
    app.config['SECRET_KEY'] = 'developing-w/o-secrets'
    app.config['MONGODB_SETTINGS'] = {
        'db': 'mnist',
        'host': 'localhost',
        'port': 27017   # default port of mongodb
    }

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    ''' initialize db connection '''
    db.init_app(app)
    print('initialized')

    ''' blueprints '''
    with app.app_context():
        from mnist import mnist
        app.register_blueprint(mnist.bp)

    return app
