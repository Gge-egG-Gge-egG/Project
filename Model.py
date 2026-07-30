import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt
#CodingMinds!!!
# Settings
dataset_path = "C:/Users/Coding Minds/Desktop/dataset2"
model_folder = "model2"
model_path = "model2/model1.keras"
class_names_path = "model2/class_names.json"
print("Working")
image_width = 224
image_height = 224
batch_size = 32
epochs = 10

# Make sure the dataset folder exists
if not os.path.exists(dataset_path):
    print("Could not find the dataset folder.")
    print("Make sure your folders are inside a folder named dataset.")
    exit()

# Create the model folder if it does not exist
os.makedirs(model_folder, exist_ok=True)

print("=" * 80)
print("LOADING RECYCLEAID DATASET")
print("=" * 80)

# Load 80% of the images for training
train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

# Load 20% of the images for validation
validation_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

# Save the class names before changing the datasets
class_names = train_data.class_names
number_of_classes = len(class_names)

print("\nClasses found:")

for class_name in class_names:
    print("-", class_name)

# Save class names in a JSON file
with open(class_names_path, "w") as file:
    json.dump(class_names, file)

# Make loading faster
train_data = train_data.prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

# Randomly modify training images
# This helps the model recognize images taken from different angles
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

print("\nLoading MobileNetV2...")

# Load a model that already understands general image features
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(image_height, image_width, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze MobileNetV2 so its existing knowledge is not overwritten
base_model.trainable = False

# Build the RecycleAid model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(image_height, image_width, 3)),
    data_augmentation,
    tf.keras.layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input
    ),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax"
    )
])

# Set up model training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

print("\n" + "=" * 60)
print("TRAINING RECYCLEAID")
print("=" * 60)

# Stop training if validation accuracy stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Save the best version during training
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path,
    monitor="val_accuracy",
    save_best_only=True
)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

validation_loss, validation_accuracy = model.evaluate(
    validation_data
)

print("Validation loss:", round(validation_loss, 4))
print(
    "Validation accuracy:",
    round(validation_accuracy * 100, 2),
    "%"
)

# Save the final trained model
model.save(model_path)

print("\nModel saved to:")
print(model_path)

print("\nClass names saved to:")
print(class_names_path)

# Graph training accuracy
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.show()

# Graph training loss
plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.show()