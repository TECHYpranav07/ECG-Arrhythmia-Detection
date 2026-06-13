import os
import cv2
import argparse

def split_13_leads_from_image(image_path, save_dir="temp_leads", auto_resize=True):
    """
    Extracts 12 standard leads + 1 long lead II rhythm strip from a standard ECG sheet.
    Expected raw resolution: 2213x1572.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")

    # Auto-resize if it doesn't match the standard dimensions
    h, w, _ = image.shape
    if (w != 2213 or h != 1572) and auto_resize:
        print(f"⚠️ Image size ({w}x{h}) does not match expected (2213x1572). Auto-resizing...")
        image = cv2.resize(image, (2213, 1572))

    os.makedirs(save_dir, exist_ok=True)

    # Crop coordinates (y1:y2, x1:x2) for 2213x1572 resolution
    Lead_1  = image[300:600, 150:643]
    Lead_2  = image[300:600, 646:1135]
    Lead_3  = image[300:600, 1140:1625]
    Lead_4  = image[300:600, 1630:2125]

    Lead_5  = image[600:900, 150:643]
    Lead_6  = image[600:900, 646:1135]
    Lead_7  = image[600:900, 1140:1625]
    Lead_8  = image[600:900, 1630:2125]

    Lead_9  = image[900:1200, 150:643]
    Lead_10 = image[900:1200, 646:1135]
    Lead_11 = image[900:1200, 1140:1625]
    Lead_12 = image[900:1200, 1630:2125]

    Lead_13 = image[1250:1480, 150:2125]  # Long Lead II

    lead_images = [
        Lead_1, Lead_2, Lead_3, Lead_4,
        Lead_5, Lead_6, Lead_7, Lead_8,
        Lead_9, Lead_10, Lead_11, Lead_12,
        Lead_13
    ]

    lead_names = [
        "01_I", "02_aVR", "03_V1", "04_V4",
        "05_II", "06_aVL", "07_V2", "08_V5",
        "09_III", "10_aVF", "11_V3", "12_V6",
        "13_Lead_II_rhythm"
    ]

    for img, name in zip(lead_images, lead_names):
        out_path = os.path.join(save_dir, f"{name}.png")
        cv2.imwrite(out_path, img)

    print(f"✅ 13 ECG leads extracted and saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 13 leads from raw ECG sheets.")
    parser.add_argument("--image", required=True, help="Path to raw ECG sheet image")
    parser.add_argument("--outdir", default="temp_leads", help="Directory to save extracted leads")
    parser.add_argument("--no-resize", action="store_false", dest="resize", help="Disable auto-resizing to 2213x1572")
    args = parser.parse_args()
    
    split_13_leads_from_image(args.image, args.outdir, args.resize)
