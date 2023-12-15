Dependencies: 
numpy 1.26.1 
Pillow 10.0.1 
PyQt5 5.15.10 
PyQt5-Qt5 5.15.11 
PyQt5-sip 12.13.0 
scipy 1.11.3 
torch 2.1.0 
torchvision 0.16.0 
pycryptodome 3.19.0 

Run demo: python demo.py 
1. Upload image(s) Click ‘upload’ button to upload an image.
2. Click ‘cancel’ to cancel uploading one image and upload a folder of images instead.
<img width="611" alt="Screenshot 2023-12-15 at 15 13 36" src="https://github.com/RhincodonE/Demo-Image-Disguising-for-Scalable-GPU-accelerated-Confidential-Deep-Learning/assets/111275412/54e0e85e-1cf9-469d-9514-9ad44c9a3053">
<img width="966" alt="Screenshot 2023-12-15 at 15 14 41" src="https://github.com/RhincodonE/Demo-Image-Disguising-for-Scalable-GPU-accelerated-Confidential-Deep-Learning/assets/111275412/64f1a4bd-cd40-4ab3-af8f-1f54266da59d">
<img width="990" alt="Screenshot 2023-12-15 at 15 15 07" src="https://github.com/RhincodonE/Demo-Image-Disguising-for-Scalable-GPU-accelerated-Confidential-Deep-Learning/assets/111275412/d57c294b-8864-4cc6-9179-7c798cb6ab92">

3. Disguised-net: Choose Block size, noise level, etc. Then click ‘Encrypt’ button to encrypt. There will pop up a dialog to choose a directory to store the encrypted image(s).
<img width="675" alt="Screenshot 2023-12-15 at 15 16 10" src="https://github.com/RhincodonE/Demo-Image-Disguising-for-Scalable-GPU-accelerated-Confidential-Deep-Learning/assets/111275412/64f22a85-6fac-4442-8621-2c84975dffce">
<img width="676" alt="Screenshot 2023-12-15 at 15 16 20" src="https://github.com/RhincodonE/Demo-Image-Disguising-for-Scalable-GPU-accelerated-Confidential-Deep-Learning/assets/111275412/1d64a142-9507-4a32-9129-b7a0415ed108">

4.Select known pairs for regression attack. Then click ‘attack’ button to do regression attack.
<img width="762" alt="Screenshot 2023-12-15 at 15 17 33" src="https://github.com/RhincodonE/Demo-Image-Disguising-for-Scalable-GPU-accelerated-Confidential-Deep-Learning/assets/111275412/5d170e80-376f-4a0b-847b-aced64e2bf4a">

5.Neuracrypt: Only need to choose block size to encrypt images. 
6.Show results button: to show experimental results in the papers.
