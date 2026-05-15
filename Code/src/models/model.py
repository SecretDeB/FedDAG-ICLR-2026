import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy 
from typing import Dict




class FcNet(nn.Module):
    """
    Fully connected network for MNIST classification
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.dims = [self.input_dim]
        self.dims.extend(hidden_dims)
        self.dims.append(self.output_dim)

        self.layers = nn.ModuleList([])

        for i in range(len(self.dims) - 1):
            ip_dim = self.dims[i]
            op_dim = self.dims[i + 1]
            self.layers.append(
                nn.Linear(ip_dim, op_dim, bias=True)
            )

        self.__init_net_weights__()

    def __init_net_weights__(self):

        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Do not apply ReLU on the final layer
            if i < (len(self.layers) - 1):
                x = F.relu(x)

            if i < (len(self.layers) - 1):  # No dropout on output layer
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x


class FCBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGGConvBlocks(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, num_classes=10):
        super(VGGConvBlocks, self).__init__()
        self.features = features
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FCBlockVGG(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(FCBlockVGG, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class SimpleCNN(nn.Module):
#     """
#     Encoder  : all conv / fc layers *except* the final classifier.
#     Classifier: 1 hidden → num_classes
#     """
#     def __init__(self, num_classes: int = 10):
#         super(SimpleCNN, self).__init__()

#         # ---------- Encoder ----------
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3,  64, 3), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(128,256, 3), nn.ReLU(), nn.MaxPool2d(2),

#             nn.Flatten(),
#             nn.Dropout(p=0.5),
#             nn.Linear(64*4*4, 128), nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(128, 256), nn.ReLU(),
#         )

#         # --------- Classifier ---------
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(256, num_classes),
#         )

#     def forward(self, x):
#         feat = self.encoder(x)
#         return self.classifier(feat)

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Encoder    : all conv/fc layers except final classifier
    Classifier : 1 hidden → num_classes
    """
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        # -------- Encoder (Conv + FC layers) --------
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),       # [B, 64, 15, 15]
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),     # [B, 128, 6, 6]
            nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),    # [B, 256, 2, 2]
            nn.Flatten(),                                                      # [B, 1024]
            nn.Dropout(p=0.5),
            nn.Linear(256 * 2 * 2, 128), nn.ReLU(),                            # → 128
            nn.Dropout(p=0.5),
            nn.Linear(128, 256), nn.ReLU(),                                    # → 256-D features
        )

        # -------- Classifier (final layer) --------
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.encoder(x)       # [B, 256]
        logits = self.classifier(features)
        return logits


import torch
import torch.nn as nn
import copy
from typing import Dict

class CombinedSimpleCNN(nn.Module):
    """
    - own_encoder        : trainable
    - secondary_encoder  : frozen
    - classifier         : on top of [own || secondary] (concatenated features)
    """
    def __init__(self,
                 own_encoder: nn.Module,
                 secondary_encoder: nn.Module,
                 classifier_sd: Dict[str, torch.Tensor],
                 num_classes: int = 10):
        super().__init__()

        # ----------------- Encoders ---------------------
        self.own_encoder = copy.deepcopy(own_encoder)
        self.secondary_encoder = copy.deepcopy(secondary_encoder)

        for p in self.secondary_encoder.parameters():
            p.requires_grad = False  # Frozen secondary

        # ----------------- Classifier --------------------
        # Each encoder outputs 256-D → concat = 512-D
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.load_state_dict(classifier_sd)

    def forward(self, x):
        f1 = self.own_encoder(x)  # Shape: [B, 256] or more
        f2 = self.secondary_encoder(x)

        # Ensure both are 2D
        f1 = f1.view(f1.size(0), -1) if f1.dim() > 2 else f1
        f2 = f2.view(f2.size(0), -1) if f2.dim() > 2 else f2

        fused = torch.cat([f1, f2], dim=1)  # Shape: [B, 512]
        return self.classifier(fused)






class SimpleCNNLight(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNLight, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
class SimpleCNN2(nn.Module):
    def __init__(self, hidden_dims, output_dim=100):  # Output_dim set to 100 for CIFAR-100
        super(SimpleCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.5)
        
        # Updated the input dimension for fc1 based on the output size of the last conv layer
        self.fc1 = nn.Linear(256 * 2 * 2, hidden_dims[0])  # 256 * 2 * 2 after conv3 and pooling
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Output size will be [batch_size, 256, 2, 2]
        x = x.view(-1, 256 * 2 * 2)  # Flatten the output
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.fc3(x)
        return x

# ================================================================
# 1.  CIFAR-10 / RGB 32×32 – 256-dim encoder
# ================================================================
class CombinedModelCIFAR(nn.Module):
    def __init__(self, own_encoder, secondary_encoder, num_classes=10):
        super().__init__()
        self.own_encoder       = copy.deepcopy(own_encoder).eval()
        self.secondary_encoder = copy.deepcopy(secondary_encoder).eval()

        for p in self.own_encoder.parameters():       p.requires_grad = False
        for p in self.secondary_encoder.parameters(): p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f1 = self.own_encoder(x)          # 256
        f2 = self.secondary_encoder(x)    # 256
        return self.classifier(torch.cat((f1, f2), dim=1))


# ================================================================
# 2.  MNIST / grayscale 28×28 – 84-dim encoder
# ================================================================

# class CombinedModelMNIST(nn.Module):
#     """
#     • own_encoder           –> trainable (fine-tuned by gradient descent)
#     • secondary_encoder     –> frozen  (kept as static feature extractor)
#     • classifier            –> trainable
#     """
#     def __init__(self, own_encoder, secondary_encoder, num_classes: int = 10):
#         super().__init__()

#         # ── 1.  clone encoders ──────────────────────────────────────────
#         self.own_encoder       = copy.deepcopy(own_encoder)        # trainable
#         self.secondary_encoder = copy.deepcopy(secondary_encoder)  # frozen

#         # Freeze ONLY the secondary encoder
#         for p in self.secondary_encoder.parameters():
#             p.requires_grad = False

#         # ── 2.  classifier on concatenated 84+84 features ──────────────
#         self.classifier = nn.Sequential(
#             nn.Linear(84 * 2, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes),
#         )

#     # ── 3.  forward pass ───────────────────────────────────────────────
#     def forward(self, x):
#         f1 = self.own_encoder(x)          # 84-D, will receive gradients
#         f2 = self.secondary_encoder(x)    # 84-D, no gradients
#         fused = torch.cat((f1, f2), dim=1)
#         return self.classifier(fused)


class CombinedModelMNIST(nn.Module):
    """
    - own_encoder        : trainable
    - secondary_encoder  : frozen
    - classifier         : on top of [own || secondary] (concatenated features)
    """
    def __init__(self,
                 own_encoder: nn.Module,
                 secondary_encoder: nn.Module,
                 classifier_sd: Dict[str, torch.Tensor],
                 num_classes: int = 10):
        super().__init__()

        # ---------- encoders -----------------------------------
        self.own_encoder       = copy.deepcopy(own_encoder)        # trainable
        self.secondary_encoder = copy.deepcopy(secondary_encoder)  # frozen
        for p in self.secondary_encoder.parameters():
            p.requires_grad = False

        # ---------- classifier on top of concatenated features -------------
        # Assuming both encoders output 84-D features → concat = 168-D
        self.classifier = nn.Linear(168, num_classes)
        self.classifier.load_state_dict(classifier_sd)

    def forward(self, x):
        f1 = self.own_encoder(x)           # Could be shape [B, 84] or [B, 84, 1, 1]
        f2 = self.secondary_encoder(x)     # Could also be 4D if not flattened internally
    
        # --- Ensure both are 2D: [batch_size, feature_dim]
        f1_flat = f1.view(f1.size(0), -1) if f1.dim() > 2 else f1
        f2_flat = f2.view(f2.size(0), -1) if f2.dim() > 2 else f2
    
        fused = torch.cat([f1_flat, f2_flat], dim=1)  # Shape: [B, 168]
        logits = self.classifier(fused)
        return logits


class CombinedSecondaryMNIST(nn.Module):
    """
    - own_encoder        : frozen (primary, not updated here)
    - secondary_encoder  : trainable (to be enriched)
    - classifier         : frozen
    """
    def __init__(self,
                 own_encoder: nn.Module,
                 secondary_encoder: nn.Module,
                 classifier_sd: Dict[str, torch.Tensor],
                 num_classes: int = 10):
        super().__init__()
        # primary (OWN) –– frozen
        self.own_encoder = copy.deepcopy(own_encoder)
        for p in self.own_encoder.parameters():
            p.requires_grad = False

        # secondary –– trainable
        self.secondary_encoder = copy.deepcopy(secondary_encoder)
        for p in self.secondary_encoder.parameters():
            p.requires_grad = True

        # classifier –– frozen
        self.classifier = nn.Linear(168, num_classes)
        self.classifier.load_state_dict(classifier_sd)
        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, x):
        f1 = self.own_encoder(x)
        f2 = self.secondary_encoder(x)
        f1 = f1.view(f1.size(0), -1) if f1.dim() > 2 else f1
        f2 = f2.view(f2.size(0), -1) if f2.dim() > 2 else f2
        return self.classifier(torch.cat([f1, f2], dim=1))


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 256)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.fc3(x)
        return x

class SimpleCNN_3(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 48, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 16 * 3 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SimpleCNNTinyImagenet_3(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNTinyImagenet_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 48, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 3 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# a simple perceptron model for generated 3D data
class PerceptronModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(PerceptronModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        x = self.fc1(x)
        return x


class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        


class SimpleCNNMNIST2(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleCNNMNIST2, self).__init__()

        # ---------- Encoder (conv + hidden FC layers) ---------------
        self.encoder = nn.Sequential(
            # conv-pool block 1
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # conv-pool block 2
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),                         # 16×5×5 = 400

            # two hidden FC layers (LeNet tradition)
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84),         nn.ReLU(),   # → 84-dim feature
        )

        # ---------- Classifier head (trainable separately) ----------
        self.classifier = nn.Linear(84, num_classes)

    # --------------------------------------------------------------
    def forward(self, x):
        features = self.encoder(x)          # 84-D embedding
        logits   = self.classifier(features)
        return logits


class SimpleCNNContainer(nn.Module):
    def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNContainer, self).__init__()
        '''
        A testing cnn container, which allows initializing a CNN with given dims

        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers

        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
        self.conv1 = nn.Conv2d(input_channel, num_filters[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


############## LeNet for MNIST ###################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = self.fc1(x)
        x = self.fc2(x)
        return x


### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNN(nn.Module):
    def __init__(self, output_dim=10):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNNCeleba(nn.Module):
    def __init__(self):
        super(ModerateCNNCeleba, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 4096)
        x = self.fc_layer(x)
        return x


class ModerateCNNMNIST(nn.Module):
    def __init__(self):
        super(ModerateCNNMNIST, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class ModerateCNNContainer(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(ModerateCNNContainer, self).__init__()

        ##
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def forward_conv(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x


