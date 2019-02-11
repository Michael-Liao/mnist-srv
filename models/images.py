from flask_mongoengine import mongoengine as ME


class Image(ME.Document):
    path = ME.StringField(required=True)
    label = ME.DecimalField(min_value=0, max_value=9)
