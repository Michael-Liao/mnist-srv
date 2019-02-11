import os
from flask import current_app, Blueprint, request, make_response
from flask_uploads import UploadSet, configure_uploads, IMAGES

from models.images import Image

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
