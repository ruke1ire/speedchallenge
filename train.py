from utils import *
from model.models import SpeedModel

def main_loop():
    snail_kwargs = {}
    vae_kwargs = {}
    device = 'cpu'

    speedmodel = SpeedModel(snail_kwargs=snail_kwargs, vae_kwargs=vae_kwargs)

try:
    main_loop()
except KeyboardInterrupt:
    print("Saving VAE model")
    torch.save(speedmodel.state_dict(), "pytorch_models/speedmodel.pt")
    sys.exit(0)

