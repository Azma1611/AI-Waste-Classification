import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def plot_samples(class_name, n_samples=20, save_name="class_samples.png"):
    folder = os.path.join("data", class_name)
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return
        
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    samples = random.sample(files, min(len(files), n_samples))
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, f in enumerate(samples):
        img_path = os.path.join(folder, f)
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f[:15], fontsize=8)
        axes[idx].axis("off")
        
    for idx in range(len(samples), len(axes)):
        axes[idx].axis("off")
        
    plt.suptitle(f"Random Samples from Category: {class_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()
    print(f"Saved visual sample grid to: {save_name}")

if __name__ == "__main__":
    os.makedirs("scratch", exist_ok=True)
    plot_samples("Plastic", save_name="scratch/plastic_samples.png")
    plot_samples("Glass", save_name="scratch/glass_samples.png")
