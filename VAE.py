import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)  
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        # For binarized (like MNIST) or grayscale images, we assume each pixel 
        # follows a Bernoulli distribution. 
        # The output physically represents the probability of the pixel being 
        # "active" (value=1). Since it is a probability, it must be strictly 
        # bounded between 0 and 1.
        """
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device).view(-1, 784)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(dataloader.dataset)
    return avg_loss

def generate_images(model, num_images, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, model.fc_mu.out_features).to(device)
        samples = model.decode(z).cpu()
    return samples.view(num_images, 1, 28, 28)

if __name__ == "__main__":
    # Example usage
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 10

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        avg_loss = train(model, train_loader, optimizer, device)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')

    # Generate sample images
    samples = generate_images(model, 16, device)
    # Here you can add code to visualize or save the generated samples

