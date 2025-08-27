For face recognition, we have to first create an enviornment using python==3.10 or higher. then install th erequirements. Note: On Windows, installing dlib sometimes fails unless you install cmake first:

pip install cmake
pip install dlib


If you don’t need dlib, you can remove it (since facenet-pytorch + mtcnn is usually enough).
then pip install -r requirements.txt

1st Step:
Create a data set like
dataset/
   person1/
      img1.jpg
      img2.jpg
   person2/
      img1.jpg
      img2.jpg
   ...

2nd Step:
Align the Images using align.py

3rd Step:
Finetune the dataset, train and create a DB to mentain the record. Use build_db.py. There is a run_command.txt for referance

4th Step:
Run main.py to recognition the face at realtime.

5th and additional Step:
if there is any changes, in dataset, life any person is removed or new person joined, just make a folder with the images, align the image, and run rebuild_db.py for update the DB.