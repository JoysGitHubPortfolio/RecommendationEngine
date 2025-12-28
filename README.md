# RecommendationEngine

**1. Identify a suitable performance/evaluation metric**
- I built a user–user collaborative filtering recommendation system using an implicit-feedback ALS model trained on historical swipe interactions to rank the top 10 profiles for each user. Model performance is evaluated using Average Precision@10 and Average Recall@10 on a held-out test set, achieving approximately 0.10 precision and 0.11 recall, which is appropriate for sparse implicit-feedback data. 

**2. Demonstrate your system working on a handful of test cases**
- The system is demonstrated on several test users, showing personalised recommendations for users with sufficient interaction history and popularity-driven recommendations for users with limited data. Fully new users are not explicitly handled in the current implementation but would be served fallback popular profiles and gradually transitioned to collaborative filtering as interactions are collected. 

**3. Briefly describe how your system would work in a production environment**
- For production, the trained model is exposed via a lightweight Flask API to support real-time inference, with clear separation between offline training and online serving, and can be deployed either via containerisation or serverless infrastructure for scalability. As users add in real-time, it may be unsustainable to do continual model re-training. Hence, some kind of batch processing may be more suitable. I would also set up alerts for issues such as data-drift.

![Systems Design](muzz_architecture_simple.png)
![Systems Design](muzz_architecture_detailed.png)
