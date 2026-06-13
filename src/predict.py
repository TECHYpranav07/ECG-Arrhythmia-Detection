import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------------------------------------------------
# Custom Capsule Network Layers (Necessary for Deserialization)
# -------------------------------------------------------------------------

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + 1e-7)

@tf.keras.utils.register_keras_serializable()
class PrimaryCaps(layers.Layer):
    def __init__(self, n_capsules, dim_capsule, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        self.n_capsules = n_capsules
        self.dim_capsule = dim_capsule
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv = layers.Conv2D(
            filters=n_capsules * dim_capsule,
            kernel_size=kernel_size,
            strides=strides,
            activation="relu",
            padding="valid"
        )

    def build(self, input_shape):
        h = (input_shape[1] - self.kernel_size) // self.strides + 1
        w = (input_shape[2] - self.kernel_size) // self.strides + 1
        self.num_caps = h * w * self.n_capsules
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = tf.reshape(x, (-1, self.num_caps, self.dim_capsule))
        return squash(x)

    def compute_output_shape(self, input_shape):
        h = (input_shape[1] - self.kernel_size) // self.strides + 1
        w = (input_shape[2] - self.kernel_size) // self.strides + 1
        return (input_shape[0], h * w * self.n_capsules, self.dim_capsule)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_capsules": self.n_capsules,
            "dim_capsule": self.dim_capsule,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class DigitCaps(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routing_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.num_input_caps = input_shape[1]
        self.input_dim = input_shape[2]
        self.W = self.add_weight(
            shape=(self.num_input_caps, self.num_capsules, self.input_dim, self.dim_capsule),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs_expanded = tf.expand_dims(inputs, 2)
        inputs_expanded = tf.expand_dims(inputs_expanded, -1)
        
        W = tf.expand_dims(self.W, 0)
        W = tf.tile(W, [batch_size, 1, 1, 1, 1])
        
        u_hat = tf.matmul(tf.transpose(W, [0,1,2,4,3]), inputs_expanded)
        u_hat = tf.squeeze(u_hat, -1)
        
        b = tf.zeros_like(tf.reduce_sum(u_hat, axis=-1))
        
        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(tf.expand_dims(c, -1) * u_hat, axis=1)
            v = squash(s)
            
            if i < self.routing_iters - 1:
                agreement = tf.reduce_sum(u_hat * tf.expand_dims(v, 1), axis=-1)
                b = b + agreement
        return v

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsule": self.dim_capsule,
            "routing_iters": self.routing_iters,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class CapsNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = models.Sequential([
            layers.Conv2D(64, (9, 9), padding="valid"),
            layers.BatchNormalization(),
            layers.Activation("relu")
        ])
        self.primary_caps = PrimaryCaps(n_capsules=8, dim_capsule=8, kernel_size=9, strides=2)
        self.digit_caps = DigitCaps(num_capsules=2, dim_capsule=16)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        length = tf.norm(x, axis=-1)
        return tf.nn.softmax(length)

# -------------------------------------------------------------------------
# Processing & Inference Functions
# -------------------------------------------------------------------------

def preprocess_lead(img_path, size=128):
    """Loads, resizes, normalizes, and shapes single lead images for CapsNet input."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    img = (img - img.mean()) / (img.std() + 1e-7)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_leads(model, lead_dir):
    """Iterates through extracted lead files and predicts abnormal/normal classifications."""
    preds = {}
    lead_files = [f for f in sorted(os.listdir(lead_dir)) if f.endswith(".png")]
    
    if not lead_files:
        raise ValueError(f"No .png lead files found in directory: {lead_dir}")
        
    for file in lead_files:
        path = os.path.join(lead_dir, file)
        x = preprocess_lead(path)
        pred = model.predict(x, verbose=0)[0]
        class_id = int(np.argmax(pred))  # 0 for abnormal, 1 for normal
        confidence = float(pred[class_id])
        preds[file] = {"class": class_id, "confidence": confidence}
        
    return preds

def weighted_vote(preds):
    """Performs majority voting across all 13 ECG leads to determine overall patient status."""
    normal_count = 0
    abnormal_count = 0
    
    for file, info in preds.items():
        if info["class"] == 1:
            normal_count += 1
        else:
            abnormal_count += 1
            
    final_class = 1 if normal_count > abnormal_count else 0
    return final_class, normal_count, abnormal_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a directory of extracted ECG leads.")
    parser.add_argument("--model", required=True, help="Path to trained Capsule Network (.h5 or keras) model file")
    parser.add_argument("--leaddir", default="temp_leads", help="Directory containing extracted .png lead images")
    args = parser.parse_args()

    # Load custom model
    print("⏳ Loading model...")
    custom_objs = {
        "PrimaryCaps": PrimaryCaps,
        "DigitCaps": DigitCaps,
        "CapsNet": CapsNet
    }
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objs)
    print("✅ Model loaded successfully!")

    # Run predictions
    print(f"⏳ Classifying leads in '{args.leaddir}'...")
    try:
        preds = predict_leads(model, args.leaddir)
        print("\n--- Lead-wise Predictions ---")
        for lead, res in preds.items():
            status = "NORMAL" if res["class"] == 1 else "ABNORMAL"
            print(f"{lead.replace('.png', '') : <20} → {status : <8} (Conf: {res['confidence']*100:.2f}%)")

        final_label, normals, abnormals = weighted_vote(preds)
        print("\n" + "="*40)
        print(f"📊 Voting Summary: Normal leads: {normals} | Abnormal leads: {abnormals}")
        print(f"诊断结论 (DIAGNOSIS): {'NORMAL' if final_label == 1 else 'ABNORMAL ARRHYTHMIA DETECTED'}")
        print("="*40)
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
