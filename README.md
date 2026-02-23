# 🏭 Warehouse Robot Intelligence System

An end-to-end applied AI system simulating a warehouse robot that:

* 📦 Detects objects using Computer Vision (OpenCV)
* 🧠 Classifies objects using a fine-tuned CNN (ResNet18)
* 📚 Retrieves grounded handling instructions using Retrieval-Augmented Generation (RAG)

This project demonstrates modular AI architecture combining perception, semantic understanding, and knowledge-grounded reasoning.

---

# 🚀 System Architecture

```
Image Input
    ↓
Vision Module (Edge + Contour Detection)
    ↓
CNN Classifier (ResNet18, Transfer Learning)
    ↓
RAG Module (FAISS + Sentence Transformers + OpenRouter)
    ↓
Grounded Handling Instructions
```

---

# 📂 Project Structure

```
warehouse_robot_ai/
├── data/
│   ├── raw/ (COCO dataset - not included in repo)
│   ├── processed/ml_dataset/
│   └── knowledge_base/
├── vision/
├── ml/
├── rag/
├── pipeline/
├── results/
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup Instructions

## 1️⃣ Create Environment

```bash
conda create -n warehouse_ai python=3.10
conda activate warehouse_ai
pip install -r requirements.txt
```

---

## 2️⃣ Download COCO Dataset

Download COCO 2017 from Kaggle and place it inside:

```
data/raw/coco2017/
```

Expected structure:

```
data/raw/coco2017/
├── train2017/
├── val2017/
└── annotations/
```

---

## 3️⃣ Configure OpenRouter API

Create a `.env` file in project root:

```
OPENROUTER_API_KEY=your_api_key_here
```

---

# 🖥️ How To Run Each Component

---

## 🔹 Part 1 — Computer Vision Module

Detect objects in an image:

```bash
python -m vision.main --image path/to/image.jpg
```

Outputs:

* Bounding boxes
* Pixel dimensions
* Center coordinates
* Annotated image saved in `results/annotated_images/`

---

## 🔹 Part 2 — Train ML Classifier

### Build Dataset from COCO

```bash
python -m ml.dataset_builder
```

### Train Model

```bash
python -m ml.train
```

### Evaluate Model

```bash
python -m ml.evaluate
```

Outputs:

* Accuracy
* Precision / Recall
* Confusion matrix saved in `results/`

---

## 🔹 Part 3 — Test RAG System

```bash
python -m rag.test_rag
```

Example queries:

* How should fragile items be handled?
* What safety checks are required for hazardous materials?
* What is the maximum lifting capacity of the gripper?

---

## 🔹 Part 4 — Full Integrated Pipeline

```bash
python -m pipeline.orchestrator
```

Workflow:

1. Detect object
2. Crop detected region
3. Classify object
4. Retrieve relevant documentation
5. Output structured result

---

# 📊 Model Performance

* Test Accuracy: **93%**
* Macro F1 Score: **0.93**
* Classes:

  * FRAGILE
  * HEAVY
  * HAZARDOUS
  * STANDARD

Class imbalance handled using weighted cross-entropy loss.

Fine-tuning of ResNet layer4 improved HEAVY recall from 0.81 → 0.92.

---

# 📚 RAG System Design

* 12 warehouse-related knowledge documents
* Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
* FAISS vector search
* Class-aware retrieval filtering
* OpenRouter LLM with strict grounding prompt

All responses are generated strictly from retrieved context to reduce hallucination risk.

---

# ⚠️ Limitations

* Vision module uses classical contour detection and may struggle in complex backgrounds.
* Dataset is derived from COCO and does not perfectly reflect real warehouse distributions.
* Classifier operates only on RGB data without depth sensing.
* Confidence outputs are not fully calibrated.
* RAG knowledge base is synthetic and limited in scope.

Future improvements could include YOLO-based detection, domain-specific data collection, multimodal sensing, and deeper LLM evaluation metrics.

---

# 🧠 Technologies Used

* Python
* OpenCV
* PyTorch
* torchvision
* FAISS
* Sentence Transformers
* OpenRouter API

---

# 📌 Challenges Faced

* Handling dataset imbalance
* Aligning classical CV outputs with CNN classifier input
* Preventing hallucination in RAG responses
* Maintaining modular project structure
* Managing package imports correctly using `python -m`

---

# 👨‍💻 Author

**VAISHAK V NAIR**

B.Tech Computer Science

AI/ML Engineer | Full-Stack Developer | Applied AI Systems Builder | LLM & Generative AI Explorer

GitHub: [https://github.com/vaishak-v-nair](https://github.com/vaishak-v-nair)
