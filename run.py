from mnist import create_app
from mnist.model import cov_net_02

if __name__ == "__main__":
    cnn = cov_net_02()
    app = create_app()
    app.run()
