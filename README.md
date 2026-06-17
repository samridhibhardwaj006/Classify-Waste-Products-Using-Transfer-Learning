# Classify-Waste-Products-Using-Transfer-Learning

📖 About

This is a Final Project from the IBM AI Engineering Professional Certificate on Coursera. It solves a real-world environmental problem faced by a fictional company, EcoClean — automating the sorting of waste using computer vision.

The project compares two approaches: Feature Extraction (frozen VGG16 layers) vs Fine-Tuning (unfreezing deeper layers), giving a hands-on comparison of both transfer learning strategies.


⏱️ Estimated time: 60 minutes




🎯 Project Objectives


✅ Apply Transfer Learning with VGG16 for binary image classification
✅ Build an Feature Extraction model (frozen base layers)
✅ Build a Fine-Tuned model (unfrozen layers, deeper training)
✅ Preprocess image data using Keras ImageDataGenerator
✅ Plot & compare loss and accuracy curves for both models
✅ Evaluate and visualize predictions on test images



🧠 Model Pipeline

Waste Image (Organic or Recyclable)
            │
            ▼
  ┌─────────────────────┐
  │  VGG16 (ImageNet)   │  ← Pretrained base
  └─────────┬───────────┘
            │
    ┌───────┴────────┐
    ▼                ▼
┌──────────────┐  ┌──────────────────┐
│  Feature     │  │   Fine-Tuning    │
│  Extraction  │  │   Model          │
│  (Frozen)    │  │  (Unfrozen)      │
└──────┬───────┘  └────────┬─────────┘
       │                   │
       ▼                   ▼
  "Organic (O)"      "Recyclable (R)"


📋 Task Breakdown

TaskDescriptionTask 1Print the TensorFlow versionTask 2Create test_generator using test_datagenTask 3Print the length of train_generatorTask 4Print the model summaryTask 5Compile the modelTask 6Plot accuracy curves — Feature Extraction modelTask 7Plot loss curves — Fine-Tuned modelTask 8Plot accuracy curves — Fine-Tuned modelTask 9Visualize test image prediction — Feature Extraction modelTask 10Visualize test image prediction — Fine-Tuned model


📂 Dataset


Source: Waste Classification Dataset — Kaggle
Classes: Organic (O) and Recyclable (R)
Split: Train / Validation (80/20) / Test
Image size: 150 × 150 px


o-vs-r-split/
├── train/
│   ├── O/    ← Organic
│   └── R/    ← Recyclable
└── test/
    ├── O/
    └── R/


🛠️ Tech Stack

ToolPurposePython 3.xCore languageTensorFlow 2.17 / KerasModel building & trainingVGG16 (ImageNet)Pretrained CNN backboneImageDataGeneratorImage loading & preprocessingScikit-learnEvaluation metricsMatplotlibLoss & accuracy visualizations


⚙️ Model Configuration

pythonimg_size    = (150, 150)
batch_size  = 32
n_epochs    = 10
n_classes   = 2
val_split   = 0.2
optimizer   = Adam
loss        = binary_crossentropy


📂 File Structure

📦 IBM-AI-Engineering-Final-Project
 ┣ 📓 Final_Proj-Classify_Waste_Products_Using_TL_FT.ipynb
 ┗ 📄 README.md


📜 Course


🎓 Final Project of the IBM AI Engineering Professional Certificate
offered by IBM on Coursera.




🙋 Author

Samridhi Bhardwaj
