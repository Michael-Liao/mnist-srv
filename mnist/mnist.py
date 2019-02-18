import os
import numpy as np
from flask import current_app, Blueprint, request, make_response
from flask_uploads import UploadSet, configure_uploads, IMAGES
from skimage import img_as_float
from skimage.io import imread

# self-defined libs
from models.images import Image
# from mnist.model import cov_net_02
import mnist

current_app.config['UPLOADED_PHOTOS_DEST'] = './data/uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(current_app, photos)

bp = Blueprint('mnist', __name__, url_prefix='/mnist')


@bp.route('/upload', methods=['POST'])
def upload():
    if 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        image = Image(
            path=os.path.join('./data/uploads', filename)
        )
        image.save()
    else:
        return make_response('no field named photo', 411)

    return 'image uploaded'


@bp.route('/predict', methods=['POST'])
def predict():
    image_path = ''
    if 'photo' in request.files:
        # upload the file to specific folder
        filename = photos.save(request.files['photo'])
        image_path = os.path.join('./data/uploads', filename)
        # update the database
        image = Image(
            path=image_path
        )
        image.save()
    else:
        return make_response('file not found', 411)

    img = imread(image_path, as_gray=True)
    img = img_as_float(img)
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)

    y = mnist.cnn.predict_on_batch(img)
    result = y[0].argmax

    print(result, type(result))

    return 'ok'
