import os
from PIL import Image, ImageDraw

def setup_mock_dataset(base_dir="data/PlantVillage", num_samples_per_class=12):
    """
    Sets up a mock PlantVillage dataset with JPEG images of healthy and diseased leaves
    for testing the data pipeline.
    """
    classes = ["Tomato_Healthy", "Tomato_Early_Blight", "Tomato_Late_Blight"]
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Setting up mock dataset in {base_dir}...")
    for class_name in classes:
        class_path = os.path.join(base_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        for i in range(num_samples_per_class):
            img_path = os.path.join(class_path, f"leaf_{i}.jpg")
            if os.path.exists(img_path):
                continue
                
            img = Image.new("RGB", (256, 256), color=(101, 67, 33))
            draw = ImageDraw.Draw(img)
            
            draw.ellipse([50, 50, 206, 206], fill=(34, 139, 34))
            
            if class_name == "Tomato_Early_Blight":
                draw.ellipse([80, 80, 100, 100], fill=(139, 69, 19))
                draw.ellipse([130, 120, 150, 140], fill=(139, 69, 19))
            elif class_name == "Tomato_Late_Blight":
                draw.ellipse([70, 70, 110, 110], fill=(80, 50, 20))
                draw.ellipse([110, 140, 160, 190], fill=(80, 50, 20))
                
            img.save(img_path, "JPEG")
            
    print("Mock dataset setup complete.")

if __name__ == "__main__":
    setup_mock_dataset()
