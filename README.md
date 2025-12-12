---
title: Boxing Dynamics Video Processor
emoji: 🎬🥊
colorFrom: blue
colorTo: blue
sdk: gradio
pinned: false
---

# Boxing Dynamics

Boxing Dynamics is a Python pipeline for analyzing the kinematics of a person boxing. The input to the pipeline is videos of person throwing a variety of punches (jab, corss, hooks or uppercuts). The pipieline will automatically analyze the video using Google's Pose Landmark Detection software [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) to detect the motion of 33 pose keypoints throughout the video. The keypoints are defined as shown in the table below. The output from the pipeline will be the input video with measured hip and shoulder rotations, as well as the hand velocity of both the left and right hands. These measurements directly correlate to an overall punch force metric. This project can be acessed online at the following huggingface space: [boxing-dynamics-demo](https://huggingface.co/spaces/adware74/boxing-dynamics-app). Alternatively, one can utilize this pipeline locally by following the usage instructions below.

### Pose Detection Landmarks
| Index | Landmark        |
|-------|------------------|
| 0     | Nose             |
| 1     | Left eye (inner) |
| 2     | Left eye         |
| 3     | Left eye (outer) |
| 4     | Right eye (inner)|
| 5     | Right eye        |
| 6     | Right eye (outer)|
| 7     | Left ear         |
| 8     | Right ear        |
| 9     | Mouth (left)     |
| 10    | Mouth (right)    |
| 11    | Left shoulder    |
| 12    | Right shoulder   |
| 13    | Left elbow       |
| 14    | Right elbow      |
| 15    | Left wrist       |
| 16    | Right wrist      |
| 17    | Left pinky       |
| 18    | Right pinky      |
| 19    | Left index       |
| 20    | Right index      |
| 21    | Left thumb       |
| 22    | Right thumb      |
| 23    | Left hip         |
| 24    | Right hip        |
| 25    | Left knee        |
| 26    | Right knee       |
| 27    | Left ankle       |
| 28    | Right ankle      |
| 29    | Left heel        |
| 30    | Right heel       |
| 31    | Left foot index  |
| 32    | Right foot index |

# Usage (PC)
In order to use the pipeline first clone the repository using : 
```
    git clone git@github.com:dhruvgargj5/boxing_dynamics.git
    cd boxing_dynamics
```

Next, create a virtual environment and install the requirements using : 
```
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
```

Finally, in order to run the pipeline use the following in terminal: 

```
    python main.py <path2Video/link2Video>
```

To run the pipeline and launch a debugger on an error:
```
python -m pdb -c continue main.py  <path2Video/link2Video>
```

# Usage (Linux/MAC)
In order to use the pipeline first clone the repository using : 
```
    git clone git@github.com:dhruvgargj5/boxing_dynamics.git
    cd boxing_dynamics
```

Next, create a virtual environment and install the requirements using : 
```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

Finally, in order to run the pipeline use the following in terminal: 
```
    python3 main.py <path2Video/link2Video>
```

To run the pipeline and launch a debugger on an error:
```
python3 -m pdb -c continue main.py  <path2Video/link2Video>
```

##
# Pipeline
The pipeline consist of 6 stages total. 
1. Video Loader 
    > loads the video from a path and stores the video settings (name, fps, and scale factor)
2. Landmark Extraction 
    > uses the mediapipe pose detection software to track the 33 keypoints throughout the video and store their positions and visibilities. 
3. Kinematics Computations 
    > computes the kinematics of relevant keypoints (left and right wrist) as well as the Joint Angular Kinematics of joints of interest using the world coordinates generated from landmark extraction 
4. Compute Boxing Metrics 
    > determines the wrist velocity and shoulder and hip rotations from the computed kinematics in the previous stage
5. Adding Force Arrows 
    > (optional stage) Adds FBD to the video which shows where the force in the punch is being generated 
6. Fusion of Boxing Metrics to Video
    > outputs the input video with additional graphs to the right of it which track the boxing metrics and the estimated kniematics from pose detection

##
# Hugging Face Deployment Guide
This guide explains how to deploy this repository as a Hugging Face Space. Follow the steps below or refer to the linked [tutorial video](https://youtu.be/8hOzsFETm4I?si=gSgO9OWUtdD0TP7i).

1. Fork the Repository
    - Create your own copy of this GitHub repository by clicking Fork.

2. Ensure Required Project Structure

    Your fork must contain:
    - main.py — runs the model pipeline or application logic
    - app.py — defines the Gradio-based user interface
    - requirements.txt — lists the Python dependencies required by the Space
    - a .gitignore which prevents large files and binary files from being pushed (.gif*, .jpeg, etc.)

3. Create a Hugging Face Account

    - Register for a free account at HuggingFace.co.

4. Create a New Hugging Face Space

    On Hugging Face:
    - Go to Spaces → New Space
    - Choose Gradio as the SDK
    - Set visibility to public
    - Create the Space

5. Create a Hugging Face Access Token

    In your Hugging Face account settings:
    - Generate a token with Write access
    - Copy and save it securely

6. Add the Token to GitHub Secrets

    In your forked GitHub repository:

    - Open Settings → Secrets and variables → Actions

    - Add a new secret:

            Name: HF_TOKEN
            Value: your Hugging Face token

7. Add the GitHub Actions Workflow
    - Create a file at .github/workflows/sync-to-huggingface.yml
    - Paste the following code block
    - Replace <hf_username> with your Hugging Face username
    - Replace <hf_space_name> with the name of your Space

```
name: Sync to Hugging Face Hub

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to Hugging Face
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git push --force \
            https://<hf_username>:$HF_TOKEN@huggingface.co/spaces/<hf_username>/<hf_space_name> \
            main
```

8. Trigger Deployment

    After committing the workflow to the main branch:
    - GitHub Actions will automatically push your repository to your Hugging Face Space.
    - Any future pushes to main will also sync automatically.

#
