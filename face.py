import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Initialize MTCNN model for single face cropping
mtcnn = MTCNN(
    image_size=160,
    margin=50,  
    min_face_size=300,  
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    keep_all=False,
    device=device
)

# Load the pre-trained model and feature extractor
extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

def detect_emotions(image):
    """
    Detect emotions from a given image.
    Returns a tuple of the cropped face image and a
    dictionary of class probabilities.
    """

    temporary = image.copy()

    # Detect faces in the image using the MTCNN group model
    sample = mtcnn.detect(temporary)
    if sample[0] is not None:
        box = sample[0][0]
        box = [int(coord) for coord in box]

        # Crop the face
        face = image.crop((box[0], box[1], box[2], box[3]))

        # Pre-process the face
        inputs = extractor(images=face, return_tensors="pt").to(device)

        # Run the image through the model
        outputs = model(**inputs)

        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Retrieve the id2label attribute from the configuration
        config = AutoConfig.from_pretrained("trpakov/vit-face-expression")
        id2label = config.id2label

        # Convert probabilities tensor to a Python list
        probabilities = probabilities.detach().cpu().numpy().tolist()[0]

        # Map class labels to their probabilities
        class_probabilities = {id2label[i]: prob for i, prob in enumerate(probabilities)}

        return face, class_probabilities
    return None, None

def plot_results(face, class_probabilities):
    if face is not None:
        # Define colors for each emotion
        colors = {
            "angry": "red",
            "disgust": "green",
            "fear": "gray",
            "happy": "yellow",
            "neutral": "purple",
            "sad": "blue",
            "surprise": "orange"
        }
        palette = [colors[label] for label in class_probabilities.keys()]

        # Create a figure with 2 subplots: one for the face image, one for the barplot
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Display face on the left subplot
        axs[0].imshow(face)
        axs[0].axis('off')

        # Create a barplot of the emotion probabilities on the right subplot
        sns.barplot(ax=axs[1], y=list(class_probabilities.keys()),
                    x=[prob * 100 for prob in class_probabilities.values()],
                    palette=palette, orient='h')
        axs[1].set_xlabel('Probability (%)')
        axs[1].set_title('Emotion Probabilities')
        axs[1].set_xlim([0, 100])

        plt.savefig("static/results.png")



def face_and_analyze():
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect emotions
        face, class_probabilities = detect_emotions(image)

        plot_results(face, class_probabilities)
        
        # Display the frame
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    face_and_analyze()
