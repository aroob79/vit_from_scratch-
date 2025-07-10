import tensorflow as tf  
from tensorflow.keras.layers import Input 
from vit_from_scratch import ViTModel   # Import ViTModel from vit_from_scratch.py 
from tensorflow.keras.models import Model 
import zipfile
from tensorflow.keras.optimizers import Adam 
import os 

IMG_SIZE = 140
NUM_CLASSES = 37

# Define the path to your ZIP file
zip_path = "archive.zip"
extract_path = "extracted_dataset_rice" 


# gpus = tf.config.list_physical_devices('GPU')
# print(f'GPUs: {gpus}')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("GPU is enabled.")
#     except RuntimeError as e:
#         print(e)

# Open the tar.gz file and extract it
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


print(f"Extracted {zip_path} to {extract_path}")



def preprocess_img(name, label):
    try:
        img = tf.io.read_file(name)
        img = tf.image.decode_jpeg(img, channels=3)
    except tf.errors.InvalidArgumentError:  # Handles corrupted or unreadable images
        print(f"Skipping corrupted image: {name}")
        return None  # Returning None to indicate removal
    
    img = tf.image.resize(img, [IMG_SIZE,IMG_SIZE])  # Resize image
    img = img / 255.0  # Normalize
    return img, label

def file_reaing(extract_path):
    images = []
    labels = []
    pathL = extract_path + '/Rice_Image_Dataset'
    for i,folder in enumerate(os.listdir(pathL)[:-1]):
        if folder.endswith('.txt'):
            continue
        print(f' reading folder {folder} started ')
        for file in os.listdir(pathL+'/'+folder): 
            images.append(pathL+'/'+folder+'/'+file)
            labels.append(i)
        print(f' reading folder {folder}  ended')   
    return images,labels


img_path,labels = file_reaing(extract_path)

# # Create a dataset without fully loading into memory
dataset = tf.data.Dataset.from_tensor_slices((img_path,labels))

# # Apply preprocessing lazily to avoid memory issues
dataset = dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)


train_ds = dataset.take(len(dataset)*0.8)
test_ds = dataset.skip(len(dataset)*0.8)

# # Shuffle, batch, and prefetch for efficiency
train_ds = train_ds.shuffle(1000).batch(40).prefetch(tf.data.AUTOTUNE) 

# # Shuffle, batch, and prefetch for efficiency
test_ds = test_ds.shuffle(1000).batch(40).prefetch(tf.data.AUTOTUNE)

PATCH_SIZE = 16

# Instantiate ViT Model
vit_model = ViTModel(
    ch=3, img_size=IMG_SIZE, patch_size=PATCH_SIZE, emb_dim=768, 
    n_layers=6, dropout=0.2, heads=4, num_classes=NUM_CLASSES
)


# Ensure TensorFlow uses GPU

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
outputs = vit_model(inputs)
model = Model(inputs, outputs)
loss  = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=Adam(learning_rate=1e-2), loss=loss, metrics=["accuracy"])
model.summary()

# Train Model
model.fit(dataset, validation_data=test_dataset, epochs=3)
print('saving the model')
model.save('train_model.keras')
# Evaluate on Test Set
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
