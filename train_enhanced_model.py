#!/usr/bin/env python3
"""
Enhanced Plant Disease Detection Model Training
Trains model with new disease classes including:
- Potato Early Blight
- Tomato Healthy
- Tomato Late Blight
- Tomato Bacterial Spot
- Tomato Target Spot
- Pepper Bell Bacterial Spot
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DATASET_DIR = 'dataset'

def create_model(num_classes):
    """Create transfer learning model with MobileNetV2"""
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def get_data_generators():
    """Create data generators with augmentation"""
    # Training data augmentation for robustness
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Validation data - only rescaling
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def save_class_names(class_indices):
    """Save class names to file"""
    # Reverse the class_indices dictionary and sort by index
    classes = sorted(class_indices.items(), key=lambda x: x[1])
    
    with open('class_names.txt', 'w') as f:
        for class_name, _ in classes:
            f.write(f"{class_name}\n")
    
    print("\n✅ Class names saved to class_names.txt")
    print("Classes:")
    for class_name, idx in classes:
        print(f"  {idx}: {class_name}")

def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-3 Accuracy if available
    if 'top_k_categorical_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_k_categorical_accuracy'], label='Training Top-3')
        axes[1, 0].plot(history.history['val_top_k_categorical_accuracy'], label='Validation Top-3')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✅ Training history plot saved to training_history.png")

def main():
    """Main training function"""
    print("=" * 60)
    print("🌱 Plant Disease Detection Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset directory '{DATASET_DIR}' not found!")
        return
    
    # Get data generators
    print("\n📂 Loading dataset...")
    train_generator, validation_generator = get_data_generators()
    
    num_classes = len(train_generator.class_indices)
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   - Total classes: {num_classes}")
    print(f"   - Training samples: {train_generator.samples}")
    print(f"   - Validation samples: {validation_generator.samples}")
    
    # Save class names
    save_class_names(train_generator.class_indices)
    
    # Create model
    print("\n🏗️  Building model architecture...")
    model, base_model = create_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_k_categorical_accuracy')]
    )
    
    print("✅ Model compiled successfully!")
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base model
    print("\n" + "=" * 60)
    print("🔥 Phase 1: Training classifier layers (base frozen)")
    print("=" * 60)
    
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune entire model
    print("\n" + "=" * 60)
    print("🔥 Phase 2: Fine-tuning entire model")
    print("=" * 60)
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_k_categorical_accuracy')]
    )
    
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS - 20,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    history = type('obj', (object,), {
        'history': {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'top_k_categorical_accuracy': history1.history['top_k_categorical_accuracy'] + history2.history['top_k_categorical_accuracy'],
            'val_top_k_categorical_accuracy': history1.history['val_top_k_categorical_accuracy'] + history2.history['val_top_k_categorical_accuracy'],
        }
    })()
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📊 Final Model Evaluation")
    print("=" * 60)
    
    final_loss, final_acc, final_top3 = model.evaluate(validation_generator, verbose=0)
    
    print(f"\n✅ Training Complete!")
    print(f"   - Final Validation Accuracy: {final_acc * 100:.2f}%")
    print(f"   - Final Validation Top-3 Accuracy: {final_top3 * 100:.2f}%")
    print(f"   - Final Validation Loss: {final_loss:.4f}")
    print(f"\n💾 Best model saved as 'best_model.h5'")
    print(f"📝 Class names saved as 'class_names.txt'")
    print(f"📈 Training history saved as 'training_history.png'")
    print("\n" + "=" * 60)

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training
    main()
