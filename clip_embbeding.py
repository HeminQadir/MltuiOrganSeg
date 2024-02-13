
import clip
import torch


## PAOT
ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 
                'Colon Tumor', 'Kidney Cyst']

print(len(ORGAN_NAME))
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device)

text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)



text_input = clip.tokenize(f'A computerized tomography scan  segment tumor').to(device)

print(text_input)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_input)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'txt_encoding.pth')