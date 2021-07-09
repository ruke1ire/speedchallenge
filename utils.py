from dojo import Dojo
import torch.nn.functional as F

class SpeedDojo(Dojo):
    def obj_func(self, recon_image, original_image, mu, logvar, pred_vel, actual_vel, vae_gain, sup_gain):
        """
        Attributes:

        recon_image = Reconstruction image from the VAE
        original_image = Original image that was fed into the VAE
        mu = Mean from VAE
        logvar = Logvar from VAE
        pred_vel = Predicted velocity from SpeedModel
        actual_vel = Actual velocity of the frame

        Returns:

        (
            total_loss = VAE loss and Supervised loss combined,
            vae_loss,
            sup_loss
        )

        """
        # VAE LOSS
        BCE = F.binary_cross_entropy(recon_image, original_image, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = BCE + KLD

        # SUPERVISED LOSS
        sup_loss = F.mse_loss(pred_vel, actual_vel)

        total_loss = vae_loss*vae_gain + sup_loss*sup_gain

        return total_loss, vae_loss, sup_loss

    def test(self, model, dataloader, logger, device):
        pass

    def train(self, model, dataloader, optimizer, device, criteria, logger):
        epoch = 1
        while True:
            for i, (image_batch, label_batch) in enumerate(dataloader):

                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                optimizer.zero_grad()

                vel_pred, vae_out = model(image_batch)

                total_loss, vae_loss, sup_loss = self.obj_func(vae_out['reconstruction'], image_batch, vae_out['mu'], vae_out['logvar'], vel_pred, label_batch, 1e-5, 1)

                total_loss.backward()
                optimizer.step()

                print(total_loss.item())
            epoch += 1

if __name__ == "__main__":
    s = SpeedDojo()

