ML Hackathon Project — HMM + Reinforcement Learning Agent for Hangman
 Overview

This project was developed in Google Colab as part of a Machine Learning Hackathon.
It combines Hidden Markov Models (HMM) and Reinforcement Learning (RL) — specifically Tabular Q-Learning and Deep Q-Network (DQN) — to play and learn the game of Hangman intelligently.
The system learns how to guess letters strategically based on word patterns, probabilities, and feedback from previous guesses.

 Model Architecture
 Hidden Markov Model (HMM)
Learns letter-to-letter transition probabilities from a large text corpus.
Predicts the probability distribution of the next letter.
Acts as a probabilistic feature extractor for the RL agent.

 Reinforcement Learning Agent
Tabular Q-Learning: Learns discrete Q-values for state-action pairs.
DQN (Deep Q-Network): Approximates the Q-function using a neural network.
Learns optimal guessing strategies through trial and error.

 Environment
Simulates the Hangman game: masked words, correct/incorrect guesses, and rewards.

 UI Dashboard

A simple Streamlit/Flask-based interface (run separately) that displays:
Current masked word
HMM letter probability predictions
RL agent’s chosen letter
Game progress and statistics

 Implementation Details
 Training Data:
Corpus size: 50,000 words
Train/Test split: 80/20

Words are preprocessed into lowercase and filtered for alphabetic characters.

 HMM Training
Trained on 40,000 words.
Transition matrix A and initial probability vector π were computed.
Model saved as hmm_model.pkl.

 RL Configuration
Parameter	Value	Description
Episodes	4000	Total training iterations
Initial ε	1.0	Pure exploration at start
ε_min	0.05	Minimum exploration
ε_decay	0.9995	Gradual shift to exploitation
Reward (Correct Guess)	+1	Positive reinforcement
Reward (Wrong Guess)	-1	Penalty for mistakes
Reward (Win)	+10	Strong positive reward
Reward (Loss)	-10	Strong negative reward

 Evaluation Results:
Training Output
Loaded 50000 words.
Train: 40000 | Test: 10000
Training HMM...
HMM fit: 100%|██████████| 40000/40000 [00:00<00:00, 151760.79it/s]
HMM saved to hmm_model.pkl
Training Q-learning agent (quick)...
Q-learn train: 100%|██████████| 4000/4000 [08:09<00:00, 8.18it/s]

 Evaluation Summary

Training Set:
{'success_rate': 0.016, 'avg_wrong': 6.97, 'avg_repeated': 0.0, 
 'avg_accuracy': 0.307, 'final_score': -17393.0}


Test Set:
{'success_rate': 0.008, 'avg_wrong': 6.99, 'avg_repeated': 0.0, 
 'avg_accuracy': 0.312, 'final_score': -17464.0}

 The agent demonstrates a basic learning trend, with moderate accuracy and limited success rate — expected due to the vast state space of Tabular Q-Learning.
A DQN-based model would improve scalability and accuracy significantly.

 Dashboard Interface
Heading: Real-Time Hangman Dashboard
The user interface was developed using Gradio in Google Colab for real-time gameplay and visualization.

Features:
Displays live word progress and letter guesses.
Visualizes HMM probabilities for each alphabet.
Shows RL agent’s decisions, rewards, and scores dynamically.
Provides interactive control — users can play alongside the agent.

🧩 Built With:

Gradio (for live dashboard)
Matplotlib and NumPy (for data visualization)
 Running the Project in Google Colab
Upload the notebook (ML_HACKATHON.ipynb) to Google Colab.
Run all cells sequentially using:
Runtime → Run all
The Gradio dashboard will automatically launch and display a shareable public link for interaction.
Training results, model checkpoints, and outputs are saved automatically in the Colab environment.

 Future Work
Replace Tabular Q-Learning with Deep Q-Network (DQN) for better scalability.
Introduce Prioritized Experience Replay (PER) to accelerate learning.
Implement reward shaping for fine-grained feedback.
Enhance Gradio UI with real-time charts and game statistics.


 Tech Stack
Language: Python
Environment: Google Colab
Libraries: NumPy, TensorFlow / PyTorch, Matplotlib, hmmlearn, tqdm, pickle, Flask / Streamlit

 Contributors
Dheeksha  — HMM Integration, data training
Rakshitha- model design, data preprocessing
chethana - reinforcement learning
shreya- ui dashboard


Team Members — UI Development and Documentation Support
