FROM python:3.7
WORKDIR /usr/app
COPY requirements.txt ./requirements.txt 
RUN pip3 install -r requirements.txt

RUN pip install opencv-python-headless
COPY ./ ./
EXPOSE 8501
CMD streamlit run streamlit_app.py