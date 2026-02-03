🚀 Amazon ML Challenge 2025: The Price is Right (Smart Pricing)
The Mission
In the fast-paced world of e-commerce, a product's success lives or dies by its price. Our team set out to build a Multi-modal Smart Pricing Engine capable of analyzing a product's visual identity (images) and its technical specifications (text) to predict the optimal price point.

The Result: A final score of 49.23 and a peak global rank of 240.

📖 Our Story: From Data to Deployment
Phase 1: The Great Gathering (Data Analysis)
We started with a massive dataset of 150,000 products. The first challenge? Noise.

The Discovery: Catalog descriptions were a "soup" of HTML, technical jargon, and Item Pack Quantities (IPQ).

The Action: We performed deep EDA and realized that price wasn't just a number; it was a relationship. A "5-pack" of batteries has a vastly different price floor than a "single" unit.

The Blueprint: We decided a simple regression wouldn't cut it. We needed to "see" the product and "read" the label simultaneously.

Phase 2: Turning Pixels into Features (Computer Vision)
We knew that high-end packaging often correlates with higher prices.

The Workflow: We benchmarked several Vision Transformers (ViT) and CNNs.

The Winner: EfficientNetB2 and InceptionResNetV2 became our workhorses.

The Artifacts: We didn't want to re-run heavy vision models during every training iteration. We performed a massive "Bake" step, extracting 150k image embeddings and saving them as .npy files for lightning-fast training later.

Phase 3: The Llama Breakthrough (NLP & LLMs)
Standard TF-IDF or Word2Vec couldn't understand the "luxury" vs "budget" nuance.

The Move: We pivoted to Llama-3 via AWS API.

The "Brain": We used the LLM to process catalog_content, extracting semantic meaning and identifying key price-driving entities (Brand, Material, Unit Count).

The Storage: These textual "thoughts" were converted into high-dimensional embeddings (text_train_emb.npy), giving our model a "human-like" understanding of the product.

Phase 4: The Fusion & The Struggle (Challenges Overcome)
This is where the real engineering happened.

Challenge - The Bottleneck: Processing 75k test images through an API is slow. We solved this using Google Colab's A100 GPUs and custom batching scripts to maximize throughput.

Challenge - Scale: Our feature matrix grew to nearly 1GB. We implemented Standard Scaling and Dimensionality Reduction to keep the model from drowning in data.

The Fusion: We joined our visual and textual vectors into a single "Master Feature Set" (X_train_scaled.npy).

Phase 5: The Final Sprint (Modeling & Submission)
With 24 hours left, we went into the "Eat" phase—rapidly iterating on the pre-computed features.

The Ensemble: We combined the strengths of Neural Networks (for complex patterns) and Gradient Boosted Trees (for structured data).

The Evaluation: Using SMAPE (Symmetric Mean Absolute Percentage Error), we tuned our hyperparameters until our local validation score mirrored the leaderboard.

The Victory: We pushed our final submission.csv, landing us in the Top 1% of the competition.

🛠️ Technical Arsenal
Component	Technology
Foundation Models	Llama 3 (Text), EfficientNet / ResNet (Vision)
Infrastructure	AWS API, Google Colab (Pro), S3
Core Libraries	TensorFlow, Keras, NumPy, Pandas, Scikit-learn
Data Format	.npy (Embeddings), .csv (Metadata)
📈 Key Lessons
Embeddings are Gold: Pre-computing features saved us 10+ hours of training time.

Multi-modal is Mandatory: Neither text nor image alone could achieve a score below 50.

Infrastructure Matters: AWS and Colab weren't just tools; they were the reason we finished on time.

Would you like me to add a "Future Improvements" section or perhaps a more detailed table of the specific hyperparameters we used for the final model?
