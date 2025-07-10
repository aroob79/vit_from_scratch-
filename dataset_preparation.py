import tensorflow as tf  
from tensorflow.keras.layers import Input 
from vit_from_scratch import ViTModel   # Import ViTModel from vit_from_scratch.py 
from tensorflow.keras.models import Model 
import tarfile
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 140
NUM_CLASSES = 37

# Define the path to your ZIP file
tar_file_path = "images.tar.gz"
extract_path = "extracted_dataset" 

# Path to the text file
file_path = "list.txt" 
path =r'extracted_dataset/images' 

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {gpus}')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is enabled.")
    except RuntimeError as e:
        print(e)

# Open the tar.gz file and extract it
with tarfile.open(tar_file_path, "r:gz") as tar:
    tar.extractall(path=extract_path)

print(f"Extracted {tar_file_path} to {extract_path}")


# Read the text file
text = tf.io.read_file(file_path).numpy().decode('utf-8')
print('file read completed')
names = [i.split()[0] for i in text.split('\n')[6:] if len(i.split()) > 0]
labels = [float(i.split()[1]) for i in text.split('\n')[6:] if len(i.split()) > 0]

train_perc = int(len(names) * 0.8)
train_names = names[:train_perc]
train_labels = labels[:train_perc] 
test_names = names[train_perc:]
test_labels = labels[train_perc:]

def preprocess_img(img_path, label):
    name = path + '/' + img_path + '.jpg'
    try:
        img = tf.io.read_file(name)
        img = tf.image.decode_jpeg(img, channels=3)
    except tf.errors.InvalidArgumentError:  # Handles corrupted or unreadable images
        print(f"Skipping corrupted image: {name}")
        return None  # Returning None to indicate removal
    
    img = tf.image.resize(img, [IMG_SIZE,IMG_SIZE])  # Resize image
    img = img / 255.0  # Normalize
    return img, label

# Create a dataset without fully loading into memory
dataset = tf.data.Dataset.from_tensor_slices((train_names, train_labels))

# Apply preprocessing lazily to avoid memory issues
dataset = dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle, batch, and prefetch for efficiency
dataset = dataset.shuffle(1000).batch(40).prefetch(tf.data.AUTOTUNE) 

# Create a dataset without fully loading into memory
test_dataset = tf.data.Dataset.from_tensor_slices((test_names, test_labels))

# Apply preprocessing lazily to avoid memory issues
test_dataset = test_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle, batch, and prefetch for efficiency
test_dataset = test_dataset.shuffle(1000).batch(40).prefetch(tf.data.AUTOTUNE)

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
