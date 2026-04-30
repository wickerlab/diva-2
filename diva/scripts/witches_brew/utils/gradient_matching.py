import torch
import torch.nn as nn
import torch.nn.functional as F

def witches_brew_optimize(victim_model, clean_images, clean_labels, target_image, target_label, epsilon=16/255, steps=100, lr=0.1):
    """
    Optimizes base images so their gradients perfectly match the gradient of a target image.
    Uses Signed-PGD bounded by an L-infinity epsilon norm.
    """
    device = next(victim_model.parameters()).device
    clean_images = clean_images.to(device)
    clean_labels = clean_labels.to(device)
    
    victim_model.eval()
    criterion = nn.CrossEntropyLoss()

    # 1. Compute the Target Gradient (The adversarial direction)
    target_image = target_image.unsqueeze(0).to(device)
    target_label = target_label.unsqueeze(0).to(device)
    
    target_loss = criterion(victim_model(target_image), target_label)
    target_grad = torch.autograd.grad(target_loss, victim_model.parameters())
    target_grad_vec = torch.cat([g.flatten() for g in target_grad]).detach()

    # Initialize poison images as a copy of clean base images
    poison_images = clean_images.clone().detach().to(device)
    
    for step in range(steps):
        poison_images.requires_grad_()
        
        # 2. Compute the Poison Gradient (Requires create_graph=True for 2nd order derivative)
        poison_loss = criterion(victim_model(poison_images), clean_labels.to(device))
        poison_grad = torch.autograd.grad(poison_loss, victim_model.parameters(), create_graph=True)
        poison_grad_vec = torch.cat([g.flatten() for g in poison_grad])
        
        # 3. Compute Cosine Similarity between the two gradients
        cos_sim = F.cosine_similarity(target_grad_vec, poison_grad_vec, dim=0)
        match_loss = 1.0 - cos_sim # We want to minimize the difference (maximize similarity)
        
        # 4. Backpropagate the matching loss back to the image pixels!
        victim_model.zero_grad()
        match_loss.backward()
        
        with torch.no_grad():
            # Apply Signed Gradient Descent
            poison_images.data = poison_images.data - lr * poison_images.grad.sign()
            
            # Project back to L-infinity Epsilon Ball to remain stealthy
            delta = torch.clamp(poison_images.data - clean_images, min=-epsilon, max=epsilon)
            poison_images.data = clean_images + delta
            
            poison_images.grad.zero_()
            
    return poison_images.detach()