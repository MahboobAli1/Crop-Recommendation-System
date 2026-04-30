from PIL import Image
import os

input_folder = "images"
output_folder = "images_resized"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(input_folder, file)
        img = Image.open(path)

        # Resize + crop to square
        img = img.resize((200, 200))

        save_path = os.path.join(output_folder, file)
        img.save(save_path)

print("All images resized!")