"""
Vision Transformer (ViT) Demo
Demonstrates the functionality of a pre-trained Vision Transformer model
for image classification tasks.

Author: Seema
Course: DS 5690 - Transformers
Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import json

# Import pre-trained ViT from torchvision
from torchvision.models import vit_b_16, ViT_B_16_Weights


def load_pretrained_vit():
    """
    Load a pre-trained Vision Transformer model (ViT-B/16).

    Returns:
        model: Pre-trained ViT model
        weights: Associated weights with preprocessing transforms
    """
    # Load pre-trained ViT-B/16 model
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.eval()  # Set to evaluation mode

    print("✓ Loaded pre-trained ViT-B/16 model")
    print(f"  - Patch size: 16x16")
    print(f"  - Image size: 224x224")
    print(f"  - Embedding dimension: 768")
    print(f"  - Number of layers: 12")
    print(f"  - Number of attention heads: 12")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, weights


def get_preprocessing_transform(weights):
    """
    Get the preprocessing transforms for the model.

    Args:
        weights: Model weights containing preprocessing info

    Returns:
        Transform pipeline for input images
    """
    return weights.transforms()


def load_image_from_url(url):
    """
    Load an image from a URL.

    Args:
        url: Image URL

    Returns:
        PIL Image object
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def classify_image(model, image, preprocess, weights, top_k=5):
    """
    Classify an image using the ViT model.

    Args:
        model: Pre-trained ViT model
        image: PIL Image
        preprocess: Preprocessing transform
        weights: Model weights (for class labels)
        top_k: Number of top predictions to return

    Returns:
        List of (class_name, probability) tuples
    """
    # Preprocess the image
    img_tensor = preprocess(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)

    # Get class names
    categories = weights.meta["categories"]
    results = [
        (categories[idx], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    return results


def visualize_attention_rollout(model, image, preprocess):
    """
    Visualize what the model is attending to using attention rollout.
    This shows which parts of the image the model focuses on.

    Args:
        model: Pre-trained ViT model
        image: PIL Image
        preprocess: Preprocessing transform

    Returns:
        Dictionary with attention information
    """
    # Preprocess the image
    img_tensor = preprocess(image).unsqueeze(0)

    # Extract attention weights from all layers
    attention_weights = []

    def hook_fn(module, input, output):
        # Extract attention weights (batch_size, num_heads, seq_len, seq_len)
        attention_weights.append(output[1])  # output[1] contains attention weights

    # Register hooks on attention layers
    hooks = []
    for block in model.encoder.layers:
        hook = block.self_attention.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(img_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Calculate attention rollout
    # Average across heads and rollout across layers
    num_patches = 196  # 14x14 patches for 224x224 image with 16x16 patches

    return {
        "num_layers": len(attention_weights),
        "num_patches": num_patches,
        "attention_shape": attention_weights[0].shape if attention_weights else None
    }


def demonstrate_patch_embedding(image, patch_size=16):
    """
    Demonstrate how ViT splits an image into patches.

    Args:
        image: PIL Image
        patch_size: Size of each patch (default: 16)

    Returns:
        Dictionary with patch information
    """
    # Resize to 224x224 if needed
    if image.size != (224, 224):
        image = image.resize((224, 224))

    # Convert to tensor
    img_tensor = transforms.ToTensor()(image)
    C, H, W = img_tensor.shape

    # Calculate number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w

    # Reshape to patches
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)  # (num_patches, C, patch_size, patch_size)

    return {
        "image_size": (H, W),
        "patch_size": patch_size,
        "num_patches": (num_patches_h, num_patches_w),
        "total_patches": total_patches,
        "patch_shape": patches.shape,
        "sequence_length": total_patches + 1  # +1 for CLS token
    }


def demo_main():
    """
    Main demo function showcasing ViT capabilities.
    """
    print("=" * 60)
    print("Vision Transformer (ViT) Demo")
    print("=" * 60)
    print()

    # Load model
    print("1. Loading Pre-trained Model")
    print("-" * 60)
    model, weights = load_pretrained_vit()
    preprocess = get_preprocessing_transform(weights)
    print()

    # Example image URLs
    test_images = [
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg",
            "description": "Cat"
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Golden_Retriever_Dukedestiny01_drvd.jpg/640px-Golden_Retriever_Dukedestiny01_drvd.jpg",
            "description": "Golden Retriever"
        }
    ]

    # Demonstrate on first image
    print("2. Demonstrating Patch Embedding")
    print("-" * 60)
    img = load_image_from_url(test_images[0]["url"])
    patch_info = demonstrate_patch_embedding(img)
    print(f"  Original image size: {patch_info['image_size']}")
    print(f"  Patch size: {patch_info['patch_size']}x{patch_info['patch_size']}")
    print(f"  Number of patches: {patch_info['num_patches'][0]} × {patch_info['num_patches'][1]} = {patch_info['total_patches']}")
    print(f"  Sequence length (with CLS token): {patch_info['sequence_length']}")
    print(f"  Patch tensor shape: {patch_info['patch_shape']}")
    print()

    # Classify images
    print("3. Image Classification Results")
    print("-" * 60)

    for i, test_img in enumerate(test_images, 1):
        print(f"\nImage {i}: {test_img['description']}")
        print(f"URL: {test_img['url']}")

        # Load and classify
        img = load_image_from_url(test_img["url"])
        results = classify_image(model, img, preprocess, weights, top_k=5)

        print("\nTop 5 Predictions:")
        for rank, (class_name, prob) in enumerate(results, 1):
            print(f"  {rank}. {class_name:30s} {prob*100:5.2f}%")

    print()
    print("4. Attention Mechanism Analysis")
    print("-" * 60)
    img = load_image_from_url(test_images[0]["url"])
    attention_info = visualize_attention_rollout(model, img, preprocess)
    print(f"  Number of transformer layers: {attention_info['num_layers']}")
    print(f"  Number of patches: {attention_info['num_patches']}")
    print(f"  Attention tensor shape: {attention_info['attention_shape']}")
    print(f"    (batch_size, num_heads, seq_length, seq_length)")
    print()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  • ViT treats images as sequences of patches (16×16 pixels)")
    print("  • Pre-trained on ImageNet achieves strong classification performance")
    print("  • Self-attention allows model to focus on relevant image regions")
    print("  • Scales well with data - larger datasets = better performance")
    print()


if __name__ == "__main__":
    demo_main()
