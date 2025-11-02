from tensorflow.keras.preprocessing.image import ImageDataGenerator

# NOTE: run this from the project root so paths resolve correctly

dg = ImageDataGenerator(rescale=1./255)
train_dir = "FER2013/train"

gen = dg.flow_from_directory(train_dir, target_size=(48,48), color_mode='grayscale', batch_size=1, class_mode='categorical')
print("class_indices:", gen.class_indices)
