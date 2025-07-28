# xG-Vision ⚽

## 🎯 Introduction

Ever wondered why a player's Expected Goals (xG) sometimes feels... off? 🤔

Standard xG models, like the fantastic one on [**Understat.com**](https://understat.com/), are revolutionary. They use sophisticated algorithms to estimate the probability of a shot becoming a goal. However, they have a crucial blind spot: they can't see the defenders or the goalkeeper blocking the path to the goal! 🥅

This project fixes that. I'm adding the missing layer of defensive context to create a more nuanced and accurate xG estimation.

## 🚀 Project Goal: Supercharging xG Calculations

I've developed a web application that analyzes video footage to extract the precise field locations of every defender and the goalkeeper at the exact moment a shot is taken. This provides the critical contextual information that current xG models are missing, leading to smarter and more realistic analytics.

## 🛠️ How It Works

My system uses cutting-edge computer vision to understand the game as it unfolds:

1. **Models** 🤖
   For this project, I trained three distinct **YOLOv8** models, which are available for direct download from Google Drive to perform the analysis::
   * One model detects **players**, **goalkeepers**, and the **referee**..
   * A second model detects the **ball**.
   * A third model identifies **keypoints** on the field (like the penalty spot, corners, etc.) using pose estimation.

2. **Goal Event Monitoring** 👀
   The system continuously tracks the ball's coordinates. When the ball crosses into the goal area, it triggers a "goal event."

3. **Shot Frame Backtracking** ⏪
   Upon detecting a goal, the system instantly backtracks through the video frames to pinpoint the exact moment the shot was taken. It's at this "shot frame" that it collects the precise positions of all players.

4. **Web Application Integration** 💻
   This entire pipeline is integrated into a slick web application. Users can upload football videos, set team names and colors, and prepare for some next-level analysis.

## ✨ Features

The application is split into two main tools:

### 1. Live Tactical Analysis
A real-time analysis tool for understanding team formations and ball movement as it happens.
* ✅ **Detects** players, referees, and the ball with high accuracy.
* 🎨 **Predicts** player teams automatically based on jersey colors.
* 🗺️ **Generates** a real-time tactical map (radar view) of player and ball positions.
* 🔍 **Tracks** every movement of the ball with a visual trail.

### 2. Goal Event Analysis
An automated tool designed to find every goal in a video and provide the critical data needed for advanced xG modeling.
* 👀 **Automated Goal Detection:** Scans the video to find every frame where the ball crosses the goal line.
* ⏪ **Shot Frame Backtracking:** For each goal, it pinpoints the moment the shot was taken by looking back a set number of frames.
* 📊 **Contextual Data Export:** For each shot leading to a goal, it extracts and displays:
    * The video frame of the shot.
    * A tactical map of all player positions.
    * The precise on-pitch coordinates (in meters) for every player.

**Current Stage:** The core model and its integration into the web application are complete. ✅

**Next Steps:** Implement the final killer feature: exporting the field locations of defenders and the goalkeeper just before a shot is taken.

## 🚀 Getting Started

Ready to run the project? Follow these simple steps to set up your local development environment.

### 1. Create a Virtual Environment

First, create a virtual environment to keep project dependencies neat and tidy. I recommend naming it `venv`.

```
python3 -m venv venv
```

### 2. Activate the Environment

Before installing packages, you need to activate the virtual environment.

**On macOS and Linux:**

```
source venv/bin/activate
```

**On Windows:**

```
.\venv\Scripts\activate
```

### 3. Install Required Packages

Install all the necessary libraries from the `requirements.txt` file.

```
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

To use the application, you must first download the trained the models yourself by running the next code.


#### a. Make the script executable (On macOS/Linux):

If you are on macOS or Linux, you first need to give the script permission to run.

```
chmod +x download_models.sh
```
#### b. Run the Download Script:

Now, run the script from your terminal. It will create a ```models/``` folder and place the files inside.


```
./download_models.sh
```

On Windows, you may need to run it with bash ```download_models.sh```.

### 5. Run the Streamlit Web application

Finally, run the app to do your video analysis.

```
streamlit run main.py
```

You are now ready to analyze football footage! 🎉 Select a demo video or upload your own, then explore the "Live Tactical Analysis" and "Goal Event Analysis" tabs.
