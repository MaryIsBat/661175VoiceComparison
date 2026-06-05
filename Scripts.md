**Original: (demo03)**

cd C:\\Users\\user\\Desktop: **Find directory to put the clone**

git clone https://github.com/resemble-ai/Resemblyzer.git Resemblyzer\_clone

cd Resemblyzer\_clone **- Go to  that directory**

py -3.10 -m venv venv **- Check python 3.10 (3.10 is required)**

.\\venv\\Scripts\\activate **- Create Virtual Environment**

pip install torch resemblyzer numpy pandas matplotlib umap-learn **- Install dependencies**
mkdir plots - **For plots**

mkdir my\_audio - **Keep your audio files here** 



**Modified version: (Voice\_gui\_optimized)**

cd C:\\Users\\Desktop

>> git clone https://github.com/resemble-ai/Resemblyzer.git Resemblyzer\_clone

>> cd Resemblyzer\_clone

>> py -3.10 -m venv venv

>> .\\venv\\Scripts\\activate

>> pip install torch resemblyzer pydub numpy umap-learn matplotlib sounddevice soundfile

>> python voice\_gui\_optimized.py

