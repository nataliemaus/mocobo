FROM haydnj/torch:bayes-lqo
# add support for molecule tasks 
RUN pip install rdkit
RUN pip install selfies 
RUN pip install guacamol 
RUN pip install scipy==1.10
# add support for Lunar Lander and Robot Pushing tasks 
RUN pip install multiprocess
RUN pip install swig
RUN pip install box2d box2d-kengz
RUN pip install gym 
RUN pip install box2d-py
# add Levenshtein for string edit distances 
RUN pip install python-Levenshtein 
#   For image tone mapping task, also add the following 
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# RUN pip install opencv-contrib-python pyiqa --force-reinstall
# NOTE: adding these dependencies for the image tone mapping task also breaks dependencies in molecule tasks 
#       so it was necessary to create a seperate Docker image used for each task