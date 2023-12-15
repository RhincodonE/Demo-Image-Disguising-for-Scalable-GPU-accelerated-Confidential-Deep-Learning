import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class NeuraCrypt(nn.Module):
    def __init__(self, image_size=(28, 28, 3), patch_size=4, noise=0, noise_level=4):
        super(NeuraCrypt, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.noise = noise
        self.noise_level = noise_level
        if len(image_size) == 3:
            self.channels = int((image_size[0]/patch_size)* (image_size[1]/patch_size)* image_size[2])
        else:
            self.channels = int((image_size[0] / patch_size) * (image_size[1] / patch_size))
        self.pos_embedding = nn.Parameter(torch.randn(self.num_patches, self.channels), requires_grad=False)

        self.encoder = self._build_encoder()
        
        if len(self.image_size) == 3:
        
            self.decoder = nn.ConvTranspose2d(self.channels, 3, kernel_size=self.patch_size, stride=self.patch_size)
        
        else:
            self.decoder = nn.ConvTranspose2d(self.channels, 1, kernel_size=self.patch_size, stride=self.patch_size)

    def _build_encoder(self):
        # Encoder with one convolutional layer
        if len(self.image_size) == 3:
        
            return nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=self.patch_size, stride=self.patch_size),
            nn.ReLU()
            )
        else:
            
            return nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=self.patch_size, stride=self.patch_size),
            nn.ReLU()
            )

    def forward(self, image):
        # Ensure the input is a PyTorch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        # Convert the image to the format [batch_size, channels, height, width] if needed
        if len(image.shape) == 3:  # Only height, width, channels
            image = image.permute(2, 0, 1).unsqueeze(0)  # Convert to [channels, height, width] and add batch dimension
        elif len(image.shape) == 4 and image.shape[1] == self.image_size[2]:  # If it's in [batch_size, channels, height, width] format
            pass  # The image is already in the correct format
        elif len(image.shape)==2:
            image = image.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Input image must have the shape [batch_size, channels, height, width], but got {image.shape}")

        # Encode image to patch features
        encoded_patches = self.encoder(image)
        
        # Permute to have patches as the last dimension
        encoded_patches = encoded_patches.flatten(2).permute(0, 2, 1)

        # Ensure that positional embeddings have the correct shape to match encoded_patches
        # We unsqueeze at 0 to add the batch dimension and expand to match the batch size of encoded_patches
        pos_embedding = self.pos_embedding.unsqueeze(0).expand(encoded_patches.size(0), -1, -1)

        # Add positional embeddings to patches
        encoded_patches += pos_embedding

        batch_size, num_patches, _ = encoded_patches.shape

        shuffle_idx = torch.randperm(num_patches).to(encoded_patches.device)

        encoded_patches = encoded_patches[:, shuffle_idx, :]

        # Reshape patches back to image grid
        H, W = self.image_size[:2]
        encoded_image = encoded_patches.permute(0, 2, 1).view(-1, self.channels, H // self.patch_size, W // self.patch_size)

        # Decode the image back to the original dimensions
        decoded_image = self.decoder(encoded_image)

        # Remove batch dimension if it was not originally there
        if decoded_image.shape[0] == 1:
            decoded_image = decoded_image.squeeze(0)
        if len(self.image_size) == 3:
            return decoded_image.permute(1,2,0)
        else:
            decoded_image = decoded_image.squeeze(0)
            return decoded_image

def test_neuracrypt():
    # Image dimensions and parameters for NeuraCrypt
    image_size = (320, 320,3)  # height, width, channels
    patch_size = 4
    noise = 1
    noise_level = 4

    # Instantiate the NeuraCrypt model
    neuracrypt = NeuraCrypt(image_size, patch_size, noise, noise_level)

    # Create a random image tensor with values between 0 and 1
    # The tensor shape should be [batch_size, channels, height, width]
    random_image = torch.rand(*image_size)

    # Run the encoding and decoding process
    encoded_image = neuracrypt.forward(random_image)
    decoded_image = neuracrypt(random_image)  # This uses the forward method
    print(encoded_image.shape)
    print(random_image.shape)

    # Check if the output dimensions match the input dimensions
    if decoded_image.size() == random_image.size():
        print("Pass: The decoded image has the same dimensions as the input image.")
    else:
        print("Fail: The decoded image dimensions do not match the input image.")




if __name__ == "__main__":

    test_neuracrypt()
