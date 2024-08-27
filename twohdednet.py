import torch
import torch.nn as nn
import torchvision.models as models

# Define the Encoder using DenseNet-121
class CustomDenseNetEncoder(nn.Module):
    def __init__(self):
        super(CustomDenseNetEncoder, self).__init__()
        # Load pre-trained DenseNet-121
        densenet = models.densenet121(pretrained=True)
        
        # Extract the features part (excluding the classifier)
        self.features = densenet.features
        
        # Replace the max-pooling layer with a 4x4 convolutional layer
        self.features.pool0 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        features = self.features(x)
        return features

# Define the Depth Estimation Decoder (DED)
class DepthEstimationDecoder(nn.Module):
    def __init__(self, input_channels):
        super(DepthEstimationDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final layer to get the depth map with 1 channel
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.decoder(x)

# Define the Deblurring Decoder (AifD)
class DeblurringDecoder(nn.Module):
    def __init__(self, input_channels):
        super(DeblurringDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final layer to get the RGB image with 3 channels
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.decoder(x)

# Define the complete 2HDED:NET model
class TwoHeadedDepthDeblurNet(nn.Module):
    def __init__(self):
        super(TwoHeadedDepthDeblurNet, self).__init__()
        # Encoder
        self.encoder = CustomDenseNetEncoder()
        
        # Depth Estimation Head (DED)
        self.depth_head = DepthEstimationDecoder(input_channels=1024)  # Adjust based on encoder output
        
        # Deblurring Head (AifD)
        self.deblurring_head = DeblurringDecoder(input_channels=1024)  # Adjust based on encoder output
        
    def forward(self, x):
        # Encode the input
        encoded_features = self.encoder(x)
        
        # Get depth map from DED
        depth_map = self.depth_head(encoded_features)
        
        # Get deblurred image from AifD
        deblurred_image = self.deblurring_head(encoded_features)
        
        return depth_map, deblurred_image

# # Example usage
# model = TwoHeadedDepthDeblurNet()
# input_tensor = torch.randn(1, 3, 224, 224)  # Example input image tensor

# depth_output, deblur_output = model(input_tensor)

# print("Depth Map Output Shape:", depth_output.shape)  # Expected: [1, 1, 224, 224]
# print("Deblurred Image Output Shape:", deblur_output.shape)  # Expected: [1, 3, 224, 224]