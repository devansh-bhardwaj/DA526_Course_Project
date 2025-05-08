import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1)
        
        # Middle
        self.mid = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1)
        
        # Decoder
        self.dec3 = nn.ConvTranspose2d(base_channels*8, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_channels*4, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(base_channels*2, out_channels, kernel_size=4, stride=2, padding=1)
        
        # Normalization layers
        self.norm1 = nn.InstanceNorm2d(base_channels)
        self.norm2 = nn.InstanceNorm2d(base_channels*2)
        self.norm4 = nn.InstanceNorm2d(base_channels*4)

    def forward(self, x):
        # Encoder
        e1 = F.leaky_relu(self.norm1(self.enc1(x)), 0.2)
        e2 = F.leaky_relu(self.norm2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.norm4(self.enc3(e2)), 0.2)

        # Middle
        mid = F.leaky_relu(self.norm4(self.mid(e3)), 0.2)

        # Decoder with skip connections
        d3 = F.leaky_relu(self.norm2(self.dec3(torch.cat([mid, e3], 1))), 0.2)
        d2 = F.leaky_relu(self.norm1(self.dec2(torch.cat([d3, e2], 1))), 0.2)
        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], 1)))

        return d1
    

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        # mask is expected to be (B, 1, H, W) initially
        # If mask is only single channel, expand to match input channels
        if mask.shape[1] == 1:
            mask = mask.expand_as(x)

        x = x * mask
        output = self.input_conv(x)
        mask_output = self.mask_conv(mask)
        mask_output = torch.clamp(mask_output, 0, 1)

        # Normalize output only where mask is non-zero
        output = output / (mask_output + 1e-8)
        output = output * mask_output

        return output, mask_output

class PartialUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder - 64x64 -> 32x32 -> 16x16 -> 8x8
        self.enc1 = PartialConv2d(in_channels, 64, 5, stride=2, padding=2)  # 64->32
        self.enc2 = PartialConv2d(64, 128, 3, stride=2, padding=1)         # 32->16
        self.enc3 = PartialConv2d(128, 256, 3, stride=2, padding=1)        # 16->8
        
        # Middle block - maintain spatial dimensions
        self.middle = PartialConv2d(256, 256, 3, stride=1, padding=1)      # 8->8
        
        # Decoder - 8x8 -> 16x16 -> 32x32 -> 64x64
        self.dec3 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)   # 8->16
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)    # 16->32
        self.dec1 = nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1) # 32->64
        
        # Normalization
        self.norm = nn.InstanceNorm2d
        self.bn256 = self.norm(256)
        self.bn128 = self.norm(128)
        self.bn64 = self.norm(64)

    def forward(self, x, mask):
        # Encoder
        e1, m1 = self.enc1(x, mask)
        e1 = F.leaky_relu(self.bn64(e1), 0.2)
        
        e2, m2 = self.enc2(e1, m1)
        e2 = F.leaky_relu(self.bn128(e2), 0.2)
        
        e3, m3 = self.enc3(e2, m2)
        e3 = F.leaky_relu(self.bn256(e3), 0.2)
        
        # Middle
        mid, _ = self.middle(e3, m3)
        mid = F.leaky_relu(self.bn256(mid), 0.2)
        
        # Decoder with skip connections
        d3 = F.leaky_relu(self.bn128(self.dec3(torch.cat([mid, e3], 1))), 0.2)
        d2 = F.leaky_relu(self.bn64(self.dec2(torch.cat([d3, e2], 1))), 0.2)
        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], 1)))
        
        return d1


class PDUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder (Reduced layers)
        self.enc1 = PartialConv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.enc2 = PartialConv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.enc3 = PartialConv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1)
        
        # Middle
        self.mid = PartialConv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1)
        
        # Decoder
        self.dec3 = nn.ConvTranspose2d(base_channels*8, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(base_channels*4, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(base_channels*2, out_channels, kernel_size=4, stride=2, padding=1)
        
        # Normalization
        self.norm1 = nn.InstanceNorm2d(base_channels)
        self.norm2 = nn.InstanceNorm2d(base_channels*2)
        self.norm4 = nn.InstanceNorm2d(base_channels*4)
        
    def forward(self, x, mask):
        # Encoder
        e1, m1 = self.enc1(x, mask)
        e1 = F.leaky_relu(self.norm1(e1), 0.2)
        
        e2, m2 = self.enc2(e1, m1)
        e2 = F.leaky_relu(self.norm2(e2), 0.2)
        
        e3, m3 = self.enc3(e2, m2)
        e3 = F.leaky_relu(self.norm4(e3), 0.2)
        
        # Middle
        mid, m_mid = self.mid(e3, m3)
        mid = F.leaky_relu(self.norm4(mid), 0.2)
        
        # Decoder with skip connections
        d3 = F.leaky_relu(self.norm2(self.dec3(torch.cat([mid, e3], 1))), 0.2)
        d2 = F.leaky_relu(self.norm1(self.dec2(torch.cat([d3, e2], 1))), 0.2)
        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], 1)))
        
        return d1
    


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),  # 64->32
            *discriminator_block(64, 128),                          # 32->16
            *discriminator_block(128, 256),                         # 16->8
            nn.Conv2d(256, 1, kernel_size=3, padding=1),           # 8->8
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

    

class StyleGANModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, style_dim=512):
        super(StyleGANModel, self).__init__()
        self.G_masked_unmasked = StyledGenerator(input_dim, hidden_dim, style_dim)
        self.G_unmasked_masked = StyledGenerator(input_dim, hidden_dim, style_dim)
        self.D_masked = StyleGANDiscriminator(input_dim, hidden_dim)
        self.D_unmasked = StyleGANDiscriminator(input_dim, hidden_dim)
        
    def forward(self, masked_images, unmasked_images):
        # Forward pass through generators
        fake_unmasked = self.G_masked_unmasked(masked_images)
        fake_masked = self.G_unmasked_masked(unmasked_images)
        
        # Cycle reconstructions
        cycle_masked = self.G_unmasked_masked(fake_unmasked)
        cycle_unmasked = self.G_masked_unmasked(fake_masked)
        
        return {
            'fake_unmasked': fake_unmasked,
            'fake_masked': fake_masked,
            'cycle_masked': cycle_masked,
            'cycle_unmasked': cycle_unmasked
        }

class StyledGenerator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, style_dim=512):
        super(StyledGenerator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, 2, 1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, 2, 1),
            nn.InstanceNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim*2, hidden_dim*4, 3, 2, 1),
            nn.InstanceNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True)
        )
        
        # Style Modulation
        self.style_dim = style_dim
        encoded_size = hidden_dim * 4 * 8 * 8  # Size after encoder
        
        self.style_modulation = nn.Sequential(
            nn.Linear(encoded_size, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, hidden_dim * 4 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, 4, 2, 1),
            nn.InstanceNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 4, 2, 1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode
        encoded = self.encoder(x)
        
        # Style modulation
        encoded_flat = encoded.view(batch_size, -1)
        styled = self.style_modulation(encoded_flat)
        styled = styled.view(batch_size, -1, 8, 8)
        
        # Decode
        output = self.decoder(styled)
        return output

class StyleGANDiscriminator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(StyleGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input layer
            nn.Conv2d(input_dim, hidden_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Hidden layers
            nn.Conv2d(hidden_dim, hidden_dim*2, 4, 2, 1),
            nn.InstanceNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim*2, hidden_dim*4, 4, 2, 1),
            nn.InstanceNorm2d(hidden_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(hidden_dim*4, 1, 4, 1, 1),
        )
        
    def forward(self, x):
        return self.model(x)
