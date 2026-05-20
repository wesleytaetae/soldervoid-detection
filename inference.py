import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==========================================
# 1. INITIALIZATION
# ==========================================
def load_production_model(weights_path: str, device: torch.device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, 
        in_channels=1,
        classes=3              
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() 
    return model

def get_inference_transforms():
    return A.Compose([
        # Fixed syntax: 'value' was deprecated in Albumentations v1.4+, use 'fill'
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, fill=0),
        ToTensorV2()
    ])

# ==========================================
# 2. METROLOGY & VISUALIZATION EXECUTION
# ==========================================
@torch.no_grad() 
def inspect_xray(model, image_path: str, device: torch.device):
    # 1. Load and normalize the raw X-ray
    raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_image is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")
        
    transform = get_inference_transforms()
    tensor_img = transform(image=raw_image)['image']
    tensor_img = tensor_img.float() / 255.0
    tensor_img = tensor_img.unsqueeze(0).to(device)
    
    # 2. The Forward Pass
    logits = model(tensor_img)
    predicted_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    
    # 3. Mathematical Area Calculation
    solder_pixels = np.sum(predicted_mask == 1)
    void_pixels = np.sum(predicted_mask == 2)
    total_area = solder_pixels + void_pixels
    
    void_ratio = 0.0
    if total_area > 0:
        void_ratio = (void_pixels / total_area) * 100.0
        
    # ==========================================
    # 4. OVERLAY GENERATION (The New Code)
    # ==========================================
    
    # Flaw Check: We cannot draw on the 'raw_image' because Albumentations 
    # might have padded it to 512x512. If we draw the 512x512 mask onto a 
    # 480x480 raw image, the coordinates will crash or misalign.
    # Solution: We extract the exact padded image from the tensor itself.
    
    # Revert the PyTorch tensor back to an 8-bit grayscale numpy array
    base_gray = (tensor_img.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    
    # Convert grayscale to a 3-channel BGR image so we can draw colored lines on it
    visual_img = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    
    # Isolate strictly the void pixels (Class 2) and convert to binary matrix
    void_binary = (predicted_mask == 2).astype(np.uint8)
    
    # Map the boundaries of the isolated void blobs
    contours, _ = cv2.findContours(void_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Paint the boundaries in bright red (BGR format: 0, 0, 255) with a thickness of 2
    cv2.drawContours(visual_img, contours, -1, (0, 0, 255), 2)
    
    # Burn the telemetry text onto the image
    text = f"Solder Void Area: {void_ratio:.2f}%"
    
    # UI Best Practice: Draw a thick black stroke first, then the green text on top.
    # This guarantees the text is readable even if it overlaps a pure white part of the X-ray.
    cv2.putText(visual_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)     # Black outline
    cv2.putText(visual_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)   # Green fill
    
    return void_ratio, predicted_mask, visual_img

# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_production_model("best_unet_model.pth", DEVICE)
    
    # Execute inspection
    ratio, mask, generated_visual = inspect_xray(model, "3.output/20.jpg", DEVICE)
    
    print(f"Inspection Complete. Ratio: {ratio:.2f}%")
    
    # Save the resulting image to your hard drive
    cv2.imwrite("inspection_report.png", generated_visual)
    print("Saved visual report to 'inspection_report.png'")