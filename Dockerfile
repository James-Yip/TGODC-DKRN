# Use the tensorflow official image as a parent image.
FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Install the required packages.
RUN pip install nltk==3.4.5 tqdm==4.36.1 thulac==0.2.0 flask \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN echo "import nltk; nltk.download('wordnet_ic')" | python

# Set the working directory.
WORKDIR /TGODC-DKRN

# Copy files from your host to your image filesystem.
COPY . .

# Install Texar locally.
RUN cd texar-0.2.1 && pip install .

# Inform Docker that the container is listening on the specified port at runtime.
EXPOSE 8080

# Run the specified command within the container.
CMD [ "python", "toc_api.py" ]
