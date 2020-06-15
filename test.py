import joblib
from utils import *
model = joblib.load('./model.pkl')
model.best_network.evaluate(episodes=1, max_episode_length=int(500), render_env=True, record=False)