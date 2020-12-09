#lightweight version---alpine
FROM python:3.7-alpine
LABEL Capstone Project

#To run python in an unbuffered mode within docker containers
#To avoid complications when running the web-app
ENV  PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

#Install dependencies
COPY requirements.txt requirements.txt


#Making a directory within docker image used to store web-app's source code.
#RUN python app.py
#EXPOSE 5000
WORKDIR /Capstone1
#COPY requirements.txt /app/
#COPY . /Capstone1
COPY . .
#RUN pip install -r requirements.txt
CMD ["python3" , "app.py"]
