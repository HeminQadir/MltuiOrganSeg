import torch
import torch.nn as nn

import torch
import torch.nn as nn

class UpsampleNetwork(nn.Module):
    def __init__(self, resolution):
        super(UpsampleNetwork, self).__init__()
        self.resolution = resolution

        # Define linear layers for initial upsampling
        self.fc1 = nn.Linear(512, resolution*resolution*resolution)
        self.prelu1 = nn.PReLU()

        # Define transpose convolution layers for upsampling
        self.conv_transpose1 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2, padding=0)
        self.prelu2 = nn.PReLU()

        self.conv_transpose2 = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2, padding=0)
        self.prelu3 = nn.PReLU()

    def forward(self, x):
        # Upsample the text embedding
        x = self.prelu1(self.fc1(x))

        print(x.shape)
        x = x.view(-1, 1, self.resolution, self.resolution, self.resolution)  # Reshape to 1x12x12x12
        
        print(x.shape)

        x = self.prelu2(self.conv_transpose1(x))  # Upsample to 1x24x24x24

        print(x.shape)
       
        x = self.prelu3(self.conv_transpose2(x)) # Upsample to 1x48x48x48

        return x


# Check for GPU and set it if available 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Create an instance of the UpsampleNetwork
resolution = 12
upsample_net = UpsampleNetwork(resolution).to(device)
print(upsample_net)
# Example usage:
input_embedding = torch.randn(1, 512).to(device)  # Input text embedding of size 1x512
output = upsample_net(input_embedding)
print(output.shape)  # Output shape should be 1x48x48x48


from torchview import draw_graph


batch_size = 2
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(upsample_net, input_size=(batch_size, 512), device='meta')
model_graph.visual_graph


print("I am done")
