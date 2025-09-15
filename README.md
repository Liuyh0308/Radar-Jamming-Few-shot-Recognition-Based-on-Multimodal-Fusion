This project proposes a deep learning framework based on multimodal fusion and meta-learning to address the few-shot recognition problem of radar active jamming signals. By fusing multimodal features such as time domain, frequency domain, and time-frequency domain, high-precision jamming recognition is achieved under the condition of limited labeled samples.

git clone git@github.com:Liuyh0308/Radar-Jamming-Few-shot-Recognition-Based-on-Multimodal-Fusion.git
cd "Radar Jamming Few-shot Recognition"

#  Create conda Env
conda create -n radar-jamming python=3.8
conda activate radar-jamming

# Install Dependents
pip install -r requirements.txt