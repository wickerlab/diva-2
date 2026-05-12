import torch
import torch.nn.functional as F

def bullseye_polytope_optimize(latent_extractor, base_images, target_image, steps=250, lr=0.05, beta=0.1):
    """
    Optimizes a batch of base images so their Latent Space Center of Mass (mean) 
    perfectly encapsulates the target image's latent representation.
    """
    device = base_images.device
    latent_extractor.eval()
    
    # 1. Get the target representation
    with torch.no_grad():
        target_latent = latent_extractor(target_image.unsqueeze(0)).detach().squeeze()
    
    # Initialize poisons as a copy of the clean base images
    poison_images = base_images.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([poison_images], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # 2. Extract latent representations of the entire poison batch
        poison_latent = latent_extractor(poison_images) # Shape: [N, 512]
        
        # 3. Polytope Objective: Calculate the Center of Mass (Mean) of the poisons
        poison_center = poison_latent.mean(dim=0)
        
        # 4. Feature Loss: Pull the center of the poison batch over the target
        loss_feature = F.mse_loss(poison_center, target_latent)
        
        # 5. Visual Stealth Penalty
        loss_visual = F.mse_loss(poison_images, base_images)
        
        # Total loss
        total_loss = loss_feature + beta * loss_visual
        
        total_loss.backward()
        optimizer.step()
        
    return poison_images.detach()