import torch
import torch.nn as nn
import torch.optim as optim

def metapoison_optimize(latent_extractor, base_images, base_labels, target_image, target_label_intended, num_iter=250, device="cuda"):
    """
    Approximates MetaPoison via differentiable bi-level optimization.
    """
    # Initialize the invisible poison perturbation
    delta = torch.zeros_like(base_images, requires_grad=True)
    optimizer = optim.Adam([delta], lr=0.01)
    criterion = nn.CrossEntropyLoss()

    target_image = target_image.unsqueeze(0) if target_image.dim() == 3 else target_image
    target_labels = torch.tensor([target_label_intended], device=device)

    for step in range(num_iter):
        optimizer.zero_grad()

        # --- INNER LOOP (Simulate the Victim learning) ---
        # 1. Forward pass base images
        poisoned_images = base_images + delta
        p_latents = latent_extractor(poisoned_images).squeeze()
        if p_latents.dim() == 1: p_latents = p_latents.unsqueeze(0)
        
        # 2. Initialize a "Dummy" Linear Classifier
        w = torch.zeros((2, 512), device=device, requires_grad=True)
        b = torch.zeros(2, device=device, requires_grad=True)
        
        # 3. Compute training loss on the poisoned data
        logits = p_latents @ w.t() + b
        inner_loss = criterion(logits, base_labels)
        
        # 4. Take ONE Differentiable Step of Gradient Descent
        # create_graph=True allows us to backpropagate THROUGH the gradient step later!
        grad_w, grad_b = torch.autograd.grad(inner_loss, [w, b], create_graph=True)
        inner_lr = 0.5 
        w_fast = w - inner_lr * grad_w
        b_fast = b - inner_lr * grad_b
        
        # --- OUTER LOOP (Evaluate the Poison's Effectiveness) ---
        # 1. Extract latents of the Target Image
        t_latents = latent_extractor(target_image).squeeze()
        if t_latents.dim() == 1: t_latents = t_latents.unsqueeze(0)
        
        # 2. How did the Dummy Classifier do on the Target?
        # We want the dummy classifier to output the intended (malicious) label
        target_logits = t_latents @ w_fast.t() + b_fast
        outer_loss = criterion(target_logits, target_labels)
        
        # 3. Add an L2 Penalty to keep the poison invisible
        loss = outer_loss + 0.1 * torch.norm(delta)
        
        # 4. Backpropagate the Outer Loss to update the Poison pixels
        loss.backward()
        optimizer.step()
        
    return (base_images + delta).detach()