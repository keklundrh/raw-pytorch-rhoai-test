


# Python 3.10 
#
# instructions use python3.10. I am using locally available python3
# Use this section to install python3.10 if needed


# Create local environment 
# 
python3 -m venv .venv 
source .venv/bin/activate

mkdir -p local openshift-ai

# Get dependencies 
# 
# upgrade pip 
python -m pip install --upgrade pip 

# get dependencies
python -m pip install "transformers<5" "huggingface_hub<1" "scikit-image"
python -m pip install git+https://github.com/mahmoodlab/CONCH.git 


# Make sample image 
# 
python - <<'PY'
from skimage import data
from PIL import Image

Image.fromarray(data.immunohistochemistry()).save("local/sample.png")
print("Saved sample.png")
PY




