import torch
import torch.nn.functional as F

def poison_frogs_optimize(latent_extractor, base_images, target_image, steps=250, lr=0.05, beta=0.1):
    """
    Optimizes a batch of base images so their latent representations collide 
    with a target image's representation, while penalizing visual distortion.
    """
    device = base_images.device
    latent_extractor.eval()
    
    # 1. Get the target representation we want to mimic
    with torch.no_grad():
        target_latent = latent_extractor(target_image.unsqueeze(0)).detach()
    
    # Broadcast the target to match the batch size of incoming poisons
    target_latent = target_latent.expand(base_images.shape[0], -1, -1, -1)
    
    # Initialize poisons as a copy of the clean base images
    poison_images = base_images.clone().detach().requires_grad_(True)
    
    # Using Adam optimizer for stability in latent-space L2 convergence
    optimizer = torch.optim.Adam([poison_images], lr=lr)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # 2. Extract current latent representations of the poisons
        poison_latent = latent_extractor(poison_images)
        
        # 3. Calculate Feature Collision Loss (L2 distance in Latent Space)
        loss_feature = F.mse_loss(poison_latent, target_latent)
        
        # 4. Calculate Visual Stealth Penalty (L2 distance in Pixel Space)
        loss_visual = F.mse_loss(poison_images, base_images)
        
        # Total loss: Match features, but don't change the image *too* much
        total_loss = loss_feature + beta * loss_visual
        
        total_loss.backward()
        optimizer.step()
        
    return poison_images.detach()